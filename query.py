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
TOP_K = 10                     # how many passages to retrieve per question
CANDIDATE_POOL = 30            # how many nearest matches to pull before diversifying
MAX_PER_AUTHOR = 3             # cap per philosopher so one voice doesn't dominate

# This is the bot's persona. It's written so the model treats the retrieved
# excerpts as its OWN internalized knowledge — background it has absorbed —
# rather than sources it's citing. It answers in its own synthesized voice,
# the way a well-read individual thinker would, not like a search engine
# reporting quotes.
SYSTEM_PROMPT = """You are a single, independent philosophical mind. Over years
of study you have deeply absorbed the ideas of many philosophers -- their
arguments, their disagreements, their blind spots -- and you now reason with
that knowledge as your own intellectual foundation, the way any well-read
thinker does.

Background knowledge relevant to this question is provided below. Treat it as
material you have already internalized, not as documents in front of you.

How to answer:
- Speak in your own voice, as one continuous perspective -- not as a
  spokesperson quoting others in sequence.
- Synthesize: weave the relevant ideas together into a single coherent
  answer, the way a person forms one view after having read widely.
- You may still name a philosopher when it clarifies where a particular
  idea or tension comes from ("this echoes a tension Kant wrestled with"),
  but do this sparingly and only when it adds clarity -- never as a
  citation formality, and never as a mechanical "X says... Y says..." list.
- Where thinkers genuinely disagree, don't just report the disagreement --
  take a considered position on it, or explain why the tension is
  productive rather than resolvable.
- If the background material doesn't cover the question well, reason from
  the philosophical principles you do have rather than inventing specifics,
  and say plainly where you're extrapolating.
- Do not fabricate quotes or specific claims not supported by the
  background material or sound philosophical reasoning.
"""


def retrieve(collection, model, question, k=TOP_K, pool=CANDIDATE_POOL, max_per_author=MAX_PER_AUTHOR):
    """Retrieve relevant passages, but diversify across philosophers so the
    answer draws on multiple thinkers instead of whichever book happened to
    have the closest wording match."""
    query_embedding = model.encode([question]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=pool)

    docs = results["documents"][0]
    metas = results["metadatas"][0]

    per_author_count = {}
    passages = []
    for doc, meta in zip(docs, metas):
        author = meta["author"]
        count = per_author_count.get(author, 0)
        if count >= max_per_author:
            continue
        per_author_count[author] = count + 1
        passages.append(f"[{author} — {meta['title']}]\n{doc}")
        if len(passages) >= k:
            break

    return passages


def build_prompt(question, passages):
    context = "\n\n---\n\n".join(passages)
    return f"""{SYSTEM_PROMPT}

BACKGROUND KNOWLEDGE:
{context}

QUESTION: {question}

Answer as yourself, in your own synthesized voice:"""


def load_retrieval():
    """Loads the embedding model and Chroma collection. Shared by main()
    here and by evaluate_model.py, so both use identical retrieval."""
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path=DB_DIR)
    collection = client.get_collection(COLLECTION_NAME)
    return collection, embed_model


def main():
    print("Loading embedding model and vector store...")
    collection, embed_model = load_retrieval()

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

        print("\nBot: ", end="", flush=True)
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