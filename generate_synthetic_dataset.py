"""
generate_synthetic_dataset.py

Generates additional training pairs for philosophers whose works are
essays/treatises rather than scripted dialogue (Kant, Mill, Hume, Marx,
etc.) -- extract_dialogue_dataset.py can't get pairs from these since
there's no speaker-labeled back-and-forth to parse.

Instead, this script uses your own LOCAL Ollama model (free, no API
costs) to read chunks of these texts and generate a plausible question
+ in-character argumentative response for each one, grounded in that
specific passage.

This is lower-quality than the real dialogue extraction (it's an LLM's
approximation of the philosopher's reasoning, not the philosopher's
actual words) so treat it as a supplement to dialogue_pairs.jsonl, not
a replacement. Spot-check the output before training on it.

Prerequisites:
    - Ollama installed and running (ollama.com)
    - A model pulled, e.g.: ollama pull llama3.1:8b

Output: dataset/synthetic_pairs.jsonl (same schema as dialogue_pairs.jsonl,
so the two files can be concatenated before fine-tuning)

Speed options (see also --help):
    python3 generate_synthetic_dataset.py --model llama3.2:3b   # smaller/faster model
    python3 generate_synthetic_dataset.py --workers 3           # parallel generation
    python3 generate_synthetic_dataset.py --max-chunks 15       # fewer chunks per book

Usage:
    python3 generate_synthetic_dataset.py
"""

import os
import re
import json
import glob
import time
import argparse
import threading
import ollama
from concurrent.futures import ThreadPoolExecutor, as_completed

TEXTS_DIR = "texts"
OUTPUT_DIR = "dataset"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "synthetic_pairs.jsonl")

DEFAULT_MODEL = "llama3.1:8b"      # match whatever you're using in query.py,
                                    # or pass a smaller/faster one with --model
                                    # e.g. llama3.2:3b or phi3

DEFAULT_CHUNK_SIZE = 1200          # characters per chunk fed to the generator
DEFAULT_MAX_CHUNKS = 40            # cap runtime -- lower with --max-chunks for a faster first run
DEFAULT_MAX_OUTPUT_TOKENS = 300    # caps generation length so calls don't run long
DEFAULT_WORKERS = 1                # how many chunks to generate concurrently

# Skip files already well covered by real dialogue extraction, since
# those already produce authentic Q&A pairs. Edit this list to match
# whichever of your texts ARE dialogues.
DIALOGUE_FILES_TO_SKIP = {"plato_republic.txt"}

GENERATION_PROMPT = """You are helping build a training dataset that captures how {author} reasons and argues, based on this excerpt from "{title}":

---
{excerpt}
---

Write ONE question a curious interlocutor might ask that this excerpt addresses, and ONE response in the voice of {author} that argues through it -- using {author}'s actual reasoning from the excerpt, not just summarizing it.

Respond with ONLY a JSON object in this exact format, nothing else:
{{"instruction": "<the question>", "output": "<{author}'s argumentative response, 2-4 sentences>"}}
"""


def parse_metadata(text):
    author_match = re.match(r"AUTHOR:\s*(.*)", text)
    title_match = re.search(r"TITLE:\s*(.*)", text)
    author = author_match.group(1).strip() if author_match else "Unknown"
    title = title_match.group(1).strip() if title_match else "Unknown"
    body = text.split("\n\n", 1)[-1]
    return author, title, body


def chunk_text(text, size):
    text = re.sub(r"\n{3,}", "\n\n", text)
    chunks = []
    start = 0
    while start < len(text):
        chunk = text[start:start + size].strip()
        if len(chunk) > 200:  # skip tiny fragments, not worth generating from
            chunks.append(chunk)
        start += size
    return chunks


def extract_json(raw_response):
    """The model sometimes wraps JSON in markdown fences or adds a little
    preamble -- pull out just the JSON object and parse it. Returns None
    if nothing valid could be extracted."""
    cleaned = raw_response.strip()
    cleaned = re.sub(r"^```(json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()

    # Find the first {...} block in case there's leading/trailing text
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        return None

    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None

    if "instruction" not in data or "output" not in data:
        return None
    if len(data["instruction"]) < 10 or len(data["output"]) < 20:
        return None

    return data


def generate_pair(chunk, author, title, model, max_output_tokens):
    prompt = GENERATION_PROMPT.format(author=author, title=title, excerpt=chunk)
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        # Capping output length keeps each call fast and predictable --
        # without this, a model can occasionally ramble well past what's
        # needed for a short JSON object.
        options={"num_predict": max_output_tokens},
    )
    raw = response["message"]["content"]
    return extract_json(raw)


def process_chunk(i, chunk, author, title, model, max_output_tokens):
    """Runs one generation and returns a result dict -- kept separate from
    the printing/writing logic so it can be called from a thread pool."""
    start_time = time.time()
    try:
        pair = generate_pair(chunk, author, title, model, max_output_tokens)
        elapsed = time.time() - start_time
        return {"index": i, "pair": pair, "elapsed": elapsed, "error": None}
    except Exception as e:
        elapsed = time.time() - start_time
        return {"index": i, "pair": None, "elapsed": elapsed, "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic Q&A training pairs via a local Ollama model.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Ollama model to use (default: {DEFAULT_MODEL}). A smaller model (e.g. llama3.2:3b) is the single biggest speed lever.")
    parser.add_argument("--max-chunks", type=int, default=DEFAULT_MAX_CHUNKS, help=f"Max chunks per book (default: {DEFAULT_MAX_CHUNKS}). Lower this for a faster run.")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE, help=f"Characters per chunk (default: {DEFAULT_CHUNK_SIZE}). Smaller chunks = less for the model to read per call.")
    parser.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS, help=f"Max generated tokens per call (default: {DEFAULT_MAX_OUTPUT_TOKENS}). Lower = faster but risks truncated JSON.")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help=f"Chunks to generate concurrently (default: {DEFAULT_WORKERS}). Only helps if your machine has spare CPU/GPU capacity -- see README for details.")
    args = parser.parse_args()

    files = [
        f for f in glob.glob(os.path.join(TEXTS_DIR, "*.txt"))
        if os.path.basename(f) not in DIALOGUE_FILES_TO_SKIP
    ]
    if not files:
        print(f"No eligible .txt files found in {TEXTS_DIR}/. Run download_texts.py first.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Using local Ollama model: {args.model}")
    if args.workers > 1:
        print(f"Running {args.workers} chunks concurrently.")
    print("Make sure Ollama is running (ollama serve) before continuing.")
    print("Each chunk requires one full model generation -- on CPU this can take")
    print("anywhere from a few seconds to over a minute per chunk. Progress prints")
    print("live below so it's clear the script is working, not stuck. Results are")
    print(f"written to {OUTPUT_FILE} incrementally, so Ctrl+C at any point keeps")
    print("whatever's been generated so far.\n")

    total_pairs = 0
    write_lock = threading.Lock()

    # Open in write mode and write as we go, rather than holding
    # everything in memory until the end -- this way an interrupted run
    # (or a crash partway through a long book) still leaves usable output.
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
        try:
            for filepath in files:
                with open(filepath, "r", encoding="utf-8") as f:
                    raw = f.read()

                author, title, body = parse_metadata(raw)
                chunks = chunk_text(body, args.chunk_size)[:args.max_chunks]

                print(f"{title} ({author}): generating from {len(chunks)} chunks...")
                book_pairs = 0

                def handle_result(result):
                    nonlocal book_pairs, total_pairs
                    i, pair, elapsed, error = result["index"], result["pair"], result["elapsed"], result["error"]

                    if error:
                        print(f"    [{i + 1}/{len(chunks)}] FAILED ({error})")
                        return
                    if pair is None:
                        print(f"    [{i + 1}/{len(chunks)}] skipped, unparseable output ({elapsed:.1f}s)")
                        return

                    record = {
                        "philosopher": author,
                        "work": title,
                        "speaker_prompt": "Interlocutor",
                        "speaker_response": author,
                        "instruction": pair["instruction"],
                        "output": pair["output"],
                    }
                    with write_lock:
                        out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                        out_f.flush()
                    book_pairs += 1
                    total_pairs += 1
                    print(f"    [{i + 1}/{len(chunks)}] OK ({elapsed:.1f}s)")

                if args.workers <= 1:
                    for i, chunk in enumerate(chunks):
                        result = process_chunk(i, chunk, author, title, args.model, args.max_output_tokens)
                        handle_result(result)
                else:
                    with ThreadPoolExecutor(max_workers=args.workers) as executor:
                        futures = [
                            executor.submit(process_chunk, i, chunk, author, title, args.model, args.max_output_tokens)
                            for i, chunk in enumerate(chunks)
                        ]
                        for future in as_completed(futures):
                            handle_result(future.result())

                print(f"  -> {book_pairs} pairs generated\n")

        except KeyboardInterrupt:
            print(f"\n\nInterrupted. Kept {total_pairs} pairs generated so far in {OUTPUT_FILE}.")
            return

    if total_pairs == 0:
        print("No pairs generated. Check that Ollama is running and the model is pulled.")
        return

    print(f"Done. Wrote {total_pairs} synthetic pairs to {OUTPUT_FILE}")
    print("This is model-generated, not verbatim source text -- spot-check a")
    print("sample for quality before combining it into your training set.")


if __name__ == "__main__":
    main()