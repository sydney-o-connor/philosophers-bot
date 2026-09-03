# Philosopher RAG Bot (100% Free, Runs Locally)

A chatbot grounded in public-domain philosophy texts (Plato, Aristotle,
Descartes, Hume, Kant, Nietzsche, Mill, Marx). No API keys, no paid
services — everything runs on your own machine.

## How it works

1. **download_texts.py** — pulls public-domain books from Project Gutenberg
2. **build_index.py** — chunks the texts and embeds them locally, storing
   them in a Chroma vector database
3. **query.py** — an interactive chat: your question gets matched against
   relevant passages from multiple philosophers, which are fed to a local
   LLM (via Ollama) as internalized background knowledge. The bot answers
   in its own synthesized voice — weaving ideas together the way a
   well-read thinker would — rather than listing citations one by one.

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

## How the bot's "voice" works

`query.py` doesn't just dump the closest-matching passage into the prompt.
Two things make it feel more like a single synthesized thinker rather than
a citation lookup:

- **Diversified retrieval**: it pulls a wider pool of candidate passages
  (30 by default) and caps how many can come from any one philosopher
  (3 by default), so an answer is more likely to actually draw on several
  thinkers instead of whichever book phrased things closest to your
  question.
- **Persona prompt**: the system prompt tells the model to treat the
  retrieved passages as knowledge it has already internalized, and to
  answer in one continuous voice — naming a philosopher only when it adds
  real clarity, not as a running citation habit.

You can tune `TOP_K`, `CANDIDATE_POOL`, and `MAX_PER_AUTHOR` at the top of
`query.py` to make answers pull from more or fewer thinkers per response.

## Notes & next steps

- **Adding more philosophers**: add entries to `BOOKS` in
  `download_texts.py`, rerun it, then rerun `build_index.py`.
- **This is still RAG under the hood**: it's a general-purpose model
  reasoning over retrieved passages, not a model that has actually
  learned to think differently. If you want a genuinely distinct
  reasoning style (not just well-briefed answers), that's a fine-tuning
  step on argumentation patterns — a bigger project on top of this one.
- **Speed**: everything runs on CPU by default. If you have a decent
  GPU, Ollama will automatically use it and responses will be much
  faster.
- **Copyright**: all texts here are public domain. If you want to add
  more recent philosophers, check their copyright status first —
  Gutenberg only hosts public-domain works.