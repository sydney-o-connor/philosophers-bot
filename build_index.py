"""
build_index.py

Reads the cleaned .txt files in ./texts/, splits them into overlapping
chunks, embeds each chunk with a free local embedding model
(sentence-transformers), and stores everything in a local Chroma
vector database (./chroma_db/).

No API keys, no internet calls other than the one-time model download.

Usage:
    python build_index.py
"""

import os
import re
import glob
import chromadb
from sentence_transformers import SentenceTransformer

TEXTS_DIR = "texts"
DB_DIR = "chroma_db"
COLLECTION_NAME = "philosophers"

CHUNK_SIZE = 800       # characters per chunk (roughly ~150-200 words)
CHUNK_OVERLAP = 150    # overlap so ideas aren't cut mid-thought


def parse_metadata(text):
    """Pull the AUTHOR / TITLE header we wrote in download_texts.py."""
    author_match = re.match(r"AUTHOR:\s*(.*)", text)
    title_match = re.search(r"TITLE:\s*(.*)", text)
    author = author_match.group(1).strip() if author_match else "Unknown"
    title = title_match.group(1).strip() if title_match else "Unknown"
    # Strip the header off before chunking
    body = text.split("\n\n", 1)[-1]
    return author, title, body


def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    text = re.sub(r"\n{3,}", "\n\n", text)  # collapse excess blank lines
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunk = text[start:end].strip()
        if len(chunk) > 50:  # skip tiny/empty fragments
            chunks.append(chunk)
        start += size - overlap
    return chunks


def main():
    files = glob.glob(os.path.join(TEXTS_DIR, "*.txt"))
    if not files:
        print(f"No .txt files found in {TEXTS_DIR}/. Run download_texts.py first.")
        return

    print("Loading local embedding model (first run downloads ~80MB, then it's cached)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    client = chromadb.PersistentClient(path=DB_DIR)
    # Fresh build each time; delete old collection if it exists
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME)

    all_chunks, all_metadatas, all_ids = [], [], []
    chunk_counter = 0

    for filepath in files:
        with open(filepath, "r", encoding="utf-8") as f:
            raw = f.read()

        author, title, body = parse_metadata(raw)
        chunks = chunk_text(body)
        print(f"  {title} ({author}): {len(chunks)} chunks")

        for chunk in chunks:
            all_chunks.append(chunk)
            all_metadatas.append({"author": author, "title": title})
            all_ids.append(f"chunk_{chunk_counter}")
            chunk_counter += 1

    print(f"\nEmbedding {len(all_chunks)} chunks locally (no API calls)...")
    embeddings = model.encode(all_chunks, show_progress_bar=True, batch_size=64)

    print("Writing to Chroma vector store...")
    # Chroma has a max batch size for add(); insert in batches to be safe
    BATCH = 500
    for i in range(0, len(all_chunks), BATCH):
        collection.add(
            documents=all_chunks[i:i + BATCH],
            embeddings=embeddings[i:i + BATCH].tolist(),
            metadatas=all_metadatas[i:i + BATCH],
            ids=all_ids[i:i + BATCH],
        )

    print(f"\nDone. Indexed {len(all_chunks)} chunks into {DB_DIR}/")


if __name__ == "__main__":
    main()