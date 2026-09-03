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

Usage:
    python3 generate_synthetic_dataset.py
"""

import os
import re
import json
import glob
import time
import ollama

TEXTS_DIR = "texts"
OUTPUT_DIR = "dataset"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "synthetic_pairs.jsonl")

OLLAMA_MODEL = "llama3.1:8b"   # match whatever you're using in query.py

CHUNK_SIZE = 1200              # characters per chunk fed to the generator
MAX_CHUNKS_PER_BOOK = 40       # cap runtime -- lower this for a faster first run
MAX_OUTPUT_TOKENS = 300        # caps generation length so calls don't run long

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


def chunk_text(text, size=CHUNK_SIZE):
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


def generate_pair(chunk, author, title):
    prompt = GENERATION_PROMPT.format(author=author, title=title, excerpt=chunk)
    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
        # Capping output length keeps each call fast and predictable --
        # without this, a model can occasionally ramble well past what's
        # needed for a short JSON object.
        options={"num_predict": MAX_OUTPUT_TOKENS},
    )
    raw = response["message"]["content"]
    return extract_json(raw)


def main():
    files = [
        f for f in glob.glob(os.path.join(TEXTS_DIR, "*.txt"))
        if os.path.basename(f) not in DIALOGUE_FILES_TO_SKIP
    ]
    if not files:
        print(f"No eligible .txt files found in {TEXTS_DIR}/. Run download_texts.py first.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Using local Ollama model: {OLLAMA_MODEL}")
    print("Make sure Ollama is running (ollama serve) before continuing.")
    print("Each chunk requires one full model generation -- on CPU this can take")
    print("anywhere from a few seconds to over a minute per chunk. Progress prints")
    print("live below so it's clear the script is working, not stuck. Results are")
    print(f"written to {OUTPUT_FILE} incrementally, so Ctrl+C at any point keeps")
    print("whatever's been generated so far.\n")

    total_pairs = 0

    # Open in append mode and write as we go, rather than holding
    # everything in memory until the end -- this way an interrupted run
    # (or a crash partway through a long book) still leaves usable output.
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
        try:
            for filepath in files:
                with open(filepath, "r", encoding="utf-8") as f:
                    raw = f.read()

                author, title, body = parse_metadata(raw)
                chunks = chunk_text(body)[:MAX_CHUNKS_PER_BOOK]

                print(f"{title} ({author}): generating from {len(chunks)} chunks...")
                book_pairs = 0

                for i, chunk in enumerate(chunks):
                    print(f"    [{i + 1}/{len(chunks)}] generating...", end=" ", flush=True)
                    start_time = time.time()

                    try:
                        pair = generate_pair(chunk, author, title)
                    except Exception as e:
                        print(f"FAILED ({e})")
                        continue

                    elapsed = time.time() - start_time

                    if pair is None:
                        print(f"skipped, unparseable output ({elapsed:.1f}s)")
                        continue

                    record = {
                        "philosopher": author,
                        "work": title,
                        "speaker_prompt": "Interlocutor",
                        "speaker_response": author,
                        "instruction": pair["instruction"],
                        "output": pair["output"],
                    }
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    out_f.flush()

                    book_pairs += 1
                    total_pairs += 1
                    print(f"OK ({elapsed:.1f}s)")

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