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

from config import DB_DIR, COLLECTION_NAME, OLLAMA_MODEL, TOP_K, CANDIDATE_POOL, MAX_PER_AUTHOR, EMBED_MODEL_NAME, MAX_HISTORY_EXCHANGES

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


def build_context_message(question, passages):
    """Formats one turn's retrieved passages + question into a user
    message. Unlike the old single-shot build_prompt(), this gets
    appended to a growing messages list rather than replacing it --
    that's what gives the bot conversation memory across turns."""
    context = "\n\n---\n\n".join(passages)
    return f"""BACKGROUND KNOWLEDGE:
{context}

QUESTION: {question}"""


def compact_message(question):
    """A stripped-down version of a past turn, used once that turn has
    already been answered. We drop the retrieved background text and
    keep just the question -- the model's own prior answer already
    reflects what it took from that background, so re-sending the raw
    passages on every later turn is pure waste. Without this, a long
    conversation's context grows by a full retrieval batch (thousands of
    tokens) every turn and can blow past the model's context window."""
    return f"QUESTION: {question}"


def trim_history(messages, max_exchanges=MAX_HISTORY_EXCHANGES):
    """Keeps the system message plus at most the last max_exchanges
    (question, answer) pairs. This is a hard backstop in addition to
    compact_message() above -- even with background text stripped,
    an unbounded number of turns would eventually still grow the
    context indefinitely, so we also cap how far back memory reaches."""
    system_msg = messages[0]
    conversation = messages[1:]
    max_messages = max_exchanges * 2  # each exchange = 1 user + 1 assistant message
    if len(conversation) > max_messages:
        conversation = conversation[-max_messages:]
    return [system_msg] + conversation


def load_retrieval():
    """Loads the embedding model and Chroma collection. Shared by main()
    here and by evaluate_model.py, so both use identical retrieval."""
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    client = chromadb.PersistentClient(path=DB_DIR)
    collection = client.get_collection(COLLECTION_NAME)
    return collection, embed_model


def main():
    print("Loading embedding model and vector store...")
    collection, embed_model = load_retrieval()

    print(f"Ready. Using Ollama model: {OLLAMA_MODEL}")
    print("Ask a philosophical question, or type 'exit' to quit.")
    print("Type 'reset' to clear conversation history and start fresh.\n")

    # This list is the bot's conversation memory -- it grows with each
    # turn so follow-up questions have the earlier exchange as context,
    # rather than each question being answered in isolation.
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    while True:
        question = input("You: ").strip()
        if question.lower() in ("exit", "quit"):
            break
        if question.lower() == "reset":
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            print("(Conversation history cleared.)\n")
            continue
        if not question:
            continue

        passages = retrieve(collection, embed_model, question)
        messages.append({"role": "user", "content": build_context_message(question, passages)})

        print("\nBot: ", end="", flush=True)
        stream = ollama.chat(
            model=OLLAMA_MODEL,
            messages=messages,
            stream=True,
        )
        answer_chunks = []
        for chunk in stream:
            piece = chunk["message"]["content"]
            answer_chunks.append(piece)
            print(piece, end="", flush=True)
        print("\n")

        # Store the assistant's reply in history too, so the NEXT turn
        # can refer back to it ("what did you just say about X?").
        messages.append({"role": "assistant", "content": "".join(answer_chunks)})

        # Strip the heavy background text out of the turn we just
        # finished (the model already used it -- keeping it around only
        # bloats every future turn's context) and cap total history
        # length as a backstop. See compact_message()/trim_history().
        messages[-2]["content"] = compact_message(question)
        messages = trim_history(messages)


if __name__ == "__main__":
    main()