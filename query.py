"""
query.py

Interactive chat loop: takes your question, retrieves the most relevant
passages from the philosopher texts (via Chroma), and asks a local
Ollama model to answer using those passages as grounding.

Prerequisites:
    1. Install Ollama: https://ollama.com
    2. Pull a model, e.g.:  ollama pull llama3.1:8b
    3. Make sure Ollama is running (it starts automatically on install,
       or run `ollama serve`)
    4. Run build_index.py first so the vector DB exists

Usage:
    python query.py
"""

import chromadb
import ollama
from sentence_transformers import SentenceTransformer

DB_DIR = "chroma_db"
COLLECTION_NAME = "philosophers"
OLLAMA_MODEL = "llama3.1:8b"   # change to any model you've pulled, e.g. "mistral"
TOP_K = 5                      # how many passages to retrieve per question

SYSTEM_PROMPT = """You are a philosophy assistant. Answer the user's question
using ONLY the excerpts provided below, which are drawn from primary texts.

Rules:
- Ground your answer in the excerpts. Reference which philosopher/work each
  idea comes from.
- If the excerpts don't contain enough to answer well, say so honestly
  rather than inventing an answer.
- Feel free to note where different philosophers in the excerpts disagree.
"""


def retrieve(collection, model, question, k=TOP_K):
    query_embedding = model.encode([question]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=k)
    passages = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        passages.append(f"[{meta['author']} — {meta['title']}]\n{doc}")
    return passages


def build_prompt(question, passages):
    context = "\n\n---\n\n".join(passages)
    return f"""{SYSTEM_PROMPT}

EXCERPTS:
{context}

QUESTION: {question}

ANSWER:"""


def main():
    print("Loading embedding model and vector store...")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path=DB_DIR)
    collection = client.get_collection(COLLECTION_NAME)

    print(f"Ready. Using Ollama model: {OLLAMA_MODEL}")
    print("Ask a philosophical question, or type 'exit' to quit.\n")

    while True:
        question = input("You: ").strip()
        if question.lower() in ("exit", "quit"):
            break
        if not question:
            continue

        passages = retrieve(collection, embed_model, question)
        prompt = build_prompt(question, passages)

        print("\nPhilosopher-bot: ", end="", flush=True)
        stream = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )
        for chunk in stream:
            print(chunk["message"]["content"], end="", flush=True)
        print("\n")


if __name__ == "__main__":
    main()
