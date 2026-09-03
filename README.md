# Philosopher RAG Bot (100% Free, Runs Locally)

A chatbot grounded in public-domain philosophy texts (Plato, Aristotle,
Descartes, Hume, Kant, Nietzsche, Mill, Marx). No API keys, no paid
services — everything runs on your own machine.

## How it works

1. **download_texts.py** — pulls public-domain books from Project Gutenberg
2. **build_index.py** — chunks the texts and embeds them locally, storing
   them in a Chroma vector database
3. **query.py** — an interactive chat: your question gets matched against
   the most relevant passages, which are fed to a local LLM (via Ollama)
   to generate a grounded answer

## Setup

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 2. Install Ollama (the free local LLM runner)

Download from **https://ollama.com** (Mac, Windows, Linux all supported).

Then pull a model — this is the actual "brain" that generates answers:

```bash
ollama pull llama3.1:8b
```

(If your machine is older/slower, try a smaller model instead, e.g.
`ollama pull mistral` or `ollama pull llama3.2:3b` — just update
`OLLAMA_MODEL` in `query.py` to match.)

### 3. Download the philosophy texts

```bash
python download_texts.py
```

This creates a `texts/` folder with cleaned `.txt` files. Feel free to
open `download_texts.py` and add more books — just find a title's "Plain
Text UTF-8" link on gutenberg.org and add a line to the `BOOKS` list.

### 4. Build the vector index

```bash
python build_index.py
```

This reads everything in `texts/`, splits it into chunks, embeds each
chunk locally, and saves it all into `chroma_db/`. Takes a few minutes
depending on your machine — it's a one-time step (rerun only if you add
more texts).

### 5. Chat with it

```bash
python query.py
```

Ask things like:
- "What does Kant say about the categorical imperative?"
- "How do Aristotle and Nietzsche differ on the idea of virtue?"
- "Summarize Mill's argument for utilitarianism"

Type `exit` to quit.

## Notes & next steps

- **Adding more philosophers**: add entries to `BOOKS` in
  `download_texts.py`, rerun it, then rerun `build_index.py`.
- **Changing the "voice"**: right now this is RAG — it retrieves and
  answers using an off-the-shelf model. If you want the bot to *sound*
  more like a specific philosopher (not just cite them), that's a
  fine-tuning step, which is a bigger project on top of this one.
- **Speed**: everything runs on CPU by default. If you have a decent
  GPU, Ollama will automatically use it and responses will be much
  faster.
- **Copyright**: all texts here are public domain. If you want to add
  more recent philosophers, check their copyright status first —
  Gutenberg only hosts public-domain works.
