"""
extract_dialogue_dataset.py

Parses speaker-labeled philosophical dialogues (Plato's works are the
best example -- Gutenberg's Jowett translations format them as
"SPEAKER. Speech text...") into question/response training pairs
suitable for fine-tuning.

This gives you REAL argumentative structure straight from the source
text, rather than synthetic data -- the model learns actual Socratic
question-and-answer patterns, not an approximation of them.

Output: dataset/dialogue_pairs.jsonl
Each line is a JSON object:
    {
      "philosopher": "Plato",
      "work": "The Republic",
      "instruction": "<the question/prompt turn>",
      "output": "<the response turn>"
    }

This JSONL format is directly usable by Hugging Face `datasets`,
Unsloth, and most fine-tuning frameworks (Alpaca-style instruction
format).

Usage:
    python3 extract_dialogue_dataset.py
"""

import os
import re
import json
import glob

TEXTS_DIR = "texts"
OUTPUT_DIR = "dataset"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "dialogue_pairs.jsonl")

MIN_RESPONSE_LEN = 40   # skip one-word acknowledgments ("True." "Yes.")
MIN_QUESTION_LEN = 10   # skip trivially short prompts

# Matches lines like "SOCRATES. ..." or "GLAUCON: ..." -- all-caps speaker
# name (1-3 words) followed by a period or colon, at the start of a line.
SPEAKER_PATTERN = re.compile(
    r"^([A-Z][A-Z]+(?:\s[A-Z][A-Z]+){0,2})[.:]\s+(.*)"
)


def parse_metadata(text):
    author_match = re.match(r"AUTHOR:\s*(.*)", text)
    title_match = re.search(r"TITLE:\s*(.*)", text)
    author = author_match.group(1).strip() if author_match else "Unknown"
    title = title_match.group(1).strip() if title_match else "Unknown"
    body = text.split("\n\n", 1)[-1]
    return author, title, body


def extract_turns(body):
    """Walk the text line by line, grouping consecutive lines that
    belong to the same speaker into a single 'turn'."""
    turns = []
    current_speaker = None
    current_lines = []

    for raw_line in body.split("\n"):
        line = raw_line.strip()
        if not line:
            continue

        match = SPEAKER_PATTERN.match(line)
        if match:
            # Flush the previous speaker's turn
            if current_speaker and current_lines:
                turns.append((current_speaker, " ".join(current_lines).strip()))
            current_speaker = match.group(1).title()
            current_lines = [match.group(2)]
        elif current_speaker:
            # Continuation of the current speaker's turn
            current_lines.append(line)

    if current_speaker and current_lines:
        turns.append((current_speaker, " ".join(current_lines).strip()))

    return turns


def build_pairs(turns, author, title):
    """Turn consecutive (speaker A, speaker B) turns into instruction/
    output training pairs. We keep it simple: any turn followed by a
    substantive response from a DIFFERENT speaker becomes a pair."""
    pairs = []
    for i in range(len(turns) - 1):
        speaker_a, text_a = turns[i]
        speaker_b, text_b = turns[i + 1]

        if speaker_a == speaker_b:
            continue
        if len(text_a) < MIN_QUESTION_LEN or len(text_b) < MIN_RESPONSE_LEN:
            continue

        pairs.append({
            "philosopher": author,
            "work": title,
            "speaker_prompt": speaker_a,
            "speaker_response": speaker_b,
            "instruction": text_a,
            "output": text_b,
        })
    return pairs


def main():
    files = glob.glob(os.path.join(TEXTS_DIR, "*.txt"))
    if not files:
        print(f"No .txt files found in {TEXTS_DIR}/. Run download_texts.py first.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_pairs = []
    for filepath in files:
        with open(filepath, "r", encoding="utf-8") as f:
            raw = f.read()

        author, title, body = parse_metadata(raw)
        turns = extract_turns(body)
        pairs = build_pairs(turns, author, title)

        if pairs:
            print(f"  {title} ({author}): {len(turns)} turns -> {len(pairs)} training pairs")
        else:
            print(f"  {title} ({author}): no speaker-labeled dialogue detected, skipping")

        all_pairs.extend(pairs)

    if not all_pairs:
        print("\nNo dialogue pairs extracted. This script works best on texts "
              "formatted as scripted dialogue (e.g. Plato's works). Non-dialogue "
              "texts (essays, treatises) won't produce pairs this way -- see the "
              "README for the synthetic-generation alternative.")
        return

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for pair in all_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"\nDone. Wrote {len(all_pairs)} training pairs to {OUTPUT_FILE}")
    print("Spot-check this file before training -- dialogue parsing from raw "
          "text is imperfect (stage directions, footnotes, or narration can "
          "slip through). Delete any bad rows by hand.")


if __name__ == "__main__":
    main()