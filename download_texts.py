"""
download_texts.py

Downloads a curated set of public-domain philosophy texts from Project
Gutenberg (www.gutenberg.org) and saves cleaned .txt files into ./texts/

All titles below are public domain (authors died 70+ years ago), so this
is legal to download and use freely, including for training/RAG.

Usage:
    python download_texts.py
"""

import os
import re
import time
import requests

OUTPUT_DIR = "texts"

# Format: (filename, philosopher, title, Gutenberg plain-text URL)
# You can add more books by finding their "Plain Text UTF-8" link on
# https://www.gutenberg.org
BOOKS = [
    ("plato_republic.txt", "Plato", "The Republic",
     "https://www.gutenberg.org/cache/epub/1497/pg1497.txt"),
    ("aristotle_ethics.txt", "Aristotle", "Nicomachean Ethics",
     "https://www.gutenberg.org/cache/epub/8438/pg8438.txt"),
    ("descartes_meditations.txt", "Descartes", "Meditations on First Philosophy",
     "https://www.gutenberg.org/cache/epub/59/pg59.txt"),
    ("hume_enquiry.txt", "Hume", "An Enquiry Concerning Human Understanding",
     "https://www.gutenberg.org/cache/epub/9662/pg9662.txt"),
    ("kant_critique.txt", "Kant", "Critique of Pure Reason",
     "https://www.gutenberg.org/cache/epub/4280/pg4280.txt"),
    ("nietzsche_zarathustra.txt", "Nietzsche", "Thus Spoke Zarathustra",
     "https://www.gutenberg.org/cache/epub/1998/pg1998.txt"),
    ("mill_utilitarianism.txt", "Mill", "Utilitarianism",
     "https://www.gutenberg.org/cache/epub/11224/pg11224.txt"),
    ("marx_manifesto.txt", "Marx & Engels", "The Communist Manifesto",
     "https://www.gutenberg.org/cache/epub/61/pg61.txt"),
     ("machiavelli_the_prince.txt", "Machiavelli", "The Prince", 
     "https://www.gutenberg.org/cache/epub/1232/pg1232.txt"),
     ("augustine_confessions.txt", "Saint Augustine", "The Confessions of St. Augustine"
     "https://www.gutenberg.org/cache/epub/3296/pg3296.txt")
]


def strip_gutenberg_boilerplate(text: str) -> str:
    """Remove the standard Gutenberg header/footer legalese, keeping only
    the actual book content."""
    start_pat = re.compile(r"\*\*\*\s*START OF (THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.IGNORECASE | re.DOTALL)
    end_pat = re.compile(r"\*\*\*\s*END OF (THE|THIS) PROJECT GUTENBERG EBOOK.*", re.IGNORECASE | re.DOTALL)

    start_match = start_pat.search(text)
    if start_match:
        text = text[start_match.end():]

    end_match = end_pat.search(text)
    if end_match:
        text = text[:end_match.start()]

    return text.strip()


def download_book(filename, author, title, url):
    dest = os.path.join(OUTPUT_DIR, filename)
    if os.path.exists(dest):
        print(f"  [skip] {title} already downloaded")
        return

    print(f"  [get]  {title} by {author} ...")
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    resp.encoding = "utf-8"

    cleaned = strip_gutenberg_boilerplate(resp.text)

    with open(dest, "w", encoding="utf-8") as f:
        # Store metadata as a header line we can parse later
        f.write(f"AUTHOR: {author}\nTITLE: {title}\n\n")
        f.write(cleaned)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Downloading {len(BOOKS)} public-domain philosophy texts...")
    for filename, author, title, url in BOOKS:
        try:
            download_book(filename, author, title, url)
        except Exception as e:
            print(f"  [FAIL] {title}: {e}")
        time.sleep(1)  # be polite to Gutenberg's servers
    print("Done. Texts saved in ./texts/")


if __name__ == "__main__":
    main()