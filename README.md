# Philosopher RAG Bot

[![Tests]([![tests.yml](https://github.com/sydney-o-connor/philosophers-bot/actions/workflows/test.yml/badge.svg)](https://github.com/sydney-o-connor/philosophers-bot/actions/workflows/test.yml))](https://github.com/sydney-o-connor/philosophers-bot/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**A local, free chatbot that synthesizes ideas from classic philosophers into one coherent voice.**

Ask it a question and it reasons across Plato, Aristotle, Descartes, Hume, Kant, Nietzsche, Mill, and Marx — weaving their ideas together like a single well-read thinker, rather than reciting quotes at you one by one. Everything runs on your own machine: no API keys, no paid services, no data leaving your laptop.

## Features

- 🧠 **Synthesized reasoning, not citation dumping** — answers in one continuous voice, naming a philosopher only when it genuinely clarifies something
- 💬 **Real conversation memory** — the chat remembers earlier turns, so follow-up questions actually build on what came before
- 🔀 **Diversified retrieval** — pulls from multiple philosophers per answer instead of whichever book happens to phrase things closest to your question
- 💸 **$0 cost** — local embeddings, local vector store, local LLM via [Ollama](https://ollama.com), and even the optional fine-tuning step uses Colab's free GPU tier
- 📚 **Public-domain sources** — texts pulled straight from Project Gutenberg
- 🎓 **Optional fine-tuning pipeline** — mine real Socratic dialogue into training data, augment with synthetic examples, and QLoRA-finetune for a more distinct reasoning style
- 📊 **Rubric-based evaluation** — automated scoring of response quality (validity, focus, groundedness, synthesis, error recovery, and more) with an LLM judge
- 🔧 **Easy to extend** — add any public-domain philosopher with one line

## Project structure

```
philosopher-rag/
├── config.py                        # shared settings (model names, DB paths) — edit here, not per-file
├── download_texts.py                # pulls public-domain texts from Project Gutenberg
├── build_index.py                   # chunks + embeds texts into a local Chroma vector DB
├── query.py                         # interactive chat loop (retrieval + synthesis prompt + memory)
├── extract_dialogue_dataset.py      # mines real Q&A pairs from Plato-style dialogues
├── generate_synthetic_dataset.py    # generates extra Q&A pairs from essay-style texts via local LLM
├── build_dataset.py                 # runs both dataset steps above + combines them (one command)
├── finetune_colab.ipynb             # free QLoRA fine-tuning notebook (Unsloth, Colab T4)
├── evaluate_model.py                # scores the bot's responses against a quality rubric
├── eval/
│   ├── rubric.md                     # human-readable rubric definitions
│   ├── test_cases.jsonl              # test conversations covering each dimension
│   └── results/                      # created by evaluate_model.py — reports land here (gitignored)
├── requirements.txt
├── .gitignore
├── LICENSE
├── tests/                            # pytest unit tests
├── texts/                             # created after download step — raw downloaded texts (gitignored)
├── dataset/                            # created by the dataset scripts — training pairs (.jsonl, gitignored)
└── chroma_db/                           # created after index-build step — the vector index (gitignored)
```

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
pip3 install -r requirements.txt
```

> On Mac/Linux, use `python3` / `pip3` throughout this README unless you've aliased `python` yourself.

### 2. Install Ollama (the free local LLM runner)

Download from **https://ollama.com** (Mac, Windows, Linux all supported).

Then pull a model — this is the actual "brain" that generates answers:

```bash
ollama pull llama3.1:8b
```

(If your machine is older/slower, try a smaller model instead, e.g.
`ollama pull mistral` or `ollama pull llama3.2:3b` — just update
`OLLAMA_MODEL` in `config.py` to match. All the scripts in this repo read
their model and path settings from `config.py`, so that's the one place
to change it.)

### 3. Download the philosophy texts

```bash
python3 download_texts.py
```

This creates a `texts/` folder with cleaned `.txt` files. Feel free to
open `download_texts.py` and add more books — just find a title's "Plain
Text UTF-8" link on gutenberg.org and add a line to the `BOOKS` list.

### 4. Build the vector index

```bash
python3 build_index.py
```

This reads everything in `texts/`, splits it into chunks, embeds each
chunk locally, and saves it all into `chroma_db/`. Takes a few minutes
depending on your machine — it's a one-time step (rerun only if you add
more texts).

### 5. Chat with it

```bash
python3 query.py
```

Ask things like:
- "What does Kant say about the categorical imperative?"
- "How do Aristotle and Nietzsche differ on the idea of virtue?"
- "Summarize Mill's argument for utilitarianism"

The chat remembers earlier turns in the conversation — you can ask a
follow-up like "but doesn't that contradict what you just said?" and it
will actually have that context. Type `reset` to clear history and start
a fresh conversation, or `exit` to quit.

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

## Going further: fine-tuning for a distinct reasoning style

RAG (everything above) gives you a general-purpose model that's well
*briefed* on philosophy — it retrieves and reasons over passages, but its
underlying reasoning process is unchanged. If you want something closer
to a genuinely distinct argumentative style, that requires fine-tuning
the model itself on examples of philosophical reasoning, not just facts.

This repo includes a free pipeline for that, using QLoRA (a lightweight
fine-tuning method) on Google Colab's free GPU tier.

### 6. Build the fine-tuning dataset (one command)

```bash
python3 build_dataset.py
```

This runs the full dataset pipeline in one step:
- **Mines real dialogue** — parses your downloaded texts for speaker-labeled
  exchanges (Plato's works are the best source — Gutenberg formats them
  as scripted dialogue) into instruction/output training pairs. This is
  **real argumentative structure straight from the source**, the
  highest-quality data in the pipeline.
- **Generates synthetic pairs** — for essay-style texts that have no
  dialogue to mine (Kant, Mill, Hume, Marx), it uses your **local** Ollama
  model (free, no API costs) to generate a plausible question + in-character
  response grounded in each passage. Lower quality than the real dialogue
  data, so it's a supplement, not a replacement.
- **Combines both** into `dataset/combined.jsonl`, ready to upload to the
  fine-tuning notebook.

If Ollama isn't running or you just want the higher-quality dialogue data
without the synthetic step:

```bash
python3 build_dataset.py --skip-synthetic
```

**Speed options**, if generation feels slow (each chunk is a full model
call — this is normal, especially on CPU):

```bash
python3 build_dataset.py --model llama3.2:3b   # smaller model = biggest speed win
python3 build_dataset.py --workers 3           # generate multiple chunks concurrently
python3 build_dataset.py --max-chunks 15       # fewer chunks per book, faster first run
```

These flags pass straight through to `generate_synthetic_dataset.py`,
which you can also run directly with `--help` to see the full set of
options (including `--chunk-size` and `--max-output-tokens`). A smaller
model (e.g. `llama3.2:3b`, pulled with `ollama pull llama3.2:3b`) is the
single biggest lever — often 3-5x faster than an 8B model on CPU, at
some cost to response coherence. `--workers` only helps if your machine
has spare CPU/GPU headroom; pushing it too high on a modest laptop can
make things slower, not faster, so try 2-3 before going higher.

**Spot-check `dataset/combined.jsonl`** before training either way — text
parsing from raw files is imperfect, and the occasional footnote,
mis-generated pair, or stage direction can slip through.

### 7. Fine-tune on Google Colab (free)

Open `finetune_colab.ipynb` in [Google Colab](https://colab.research.google.com):

1. `Runtime` → `Change runtime type` → select **T4 GPU** (free tier)
2. Upload your `dataset/combined.jsonl` using the Colab file browser
3. Run all cells top to bottom

The notebook uses [Unsloth](https://github.com/unslothai/unsloth) to
QLoRA-finetune Llama 3.1 8B on your dataset — small trainable adapter
layers on top of a frozen base model, which is what makes this feasible
on a free GPU. Training takes roughly 30–90 minutes depending on dataset
size. It ends by exporting a GGUF file you can run locally.

This step can't be folded into `build_dataset.py` — Colab needs you to
manually pick a GPU runtime and upload the file through its UI.

### 8. Run your fine-tuned model locally with Ollama

After downloading the `.gguf` file from Colab:

```bash
echo "FROM ./your-model.gguf" > Modelfile
ollama create philosopher-custom -f Modelfile
```

Then update `OLLAMA_MODEL = "philosopher-custom"` in `query.py`. Everything
else — retrieval, the persona prompt, the chat loop — works unchanged.

### Honest expectations

- Dataset size drives quality more than anything else. A few hundred
  dialogue pairs will give you a subtle style shift, not a dramatic
  transformation — that's normal for this scale of fine-tuning.
- This changes *style and reasoning rhythm*, not factual knowledge — RAG
  is still doing the heavy lifting on content grounding.
- Colab's free tier can disconnect on idle or high GPU demand — don't
  close the tab mid-training.

## Testing the bot's responses (evaluation rubric)

Beyond unit tests for the scripts themselves, this repo includes an
automated rubric harness for judging the *quality* of the bot's actual
answers — separate from whether the code runs correctly.

```bash
python3 evaluate_model.py
```

This runs a set of test conversations from `eval/test_cases.jsonl`
through your bot (using the same retrieval + persona setup as `query.py`),
then uses an LLM judge to score each response 1-5 across seven
dimensions:

- **Validity** — is the reasoning logically sound?
- **Focus** — does it actually answer what was asked?
- **Groundedness** — are claims real, not fabricated?
- **Synthesis** — one coherent voice, or a citation list?
- **Progress** *(multi-turn only)* — does it build on the conversation?
- **Error recovery** — does it admit gaps instead of confabulating on
  out-of-scope or trick questions?
- **Voice consistency** — does the persona hold up across a conversation?

Full definitions are in `eval/rubric.md`. The test set deliberately
includes adversarial cases — questions about philosophers not in your
corpus, requests for fabricated quotes, an instruction-override attempt
— specifically to exercise error recovery, since easy questions can't
test that dimension.

Output lands in `eval/results/`: a markdown report with per-dimension
averages and any case that scored 1-2 on something flagged for review,
plus the raw JSON scores.

**Useful flags:**

```bash
python3 evaluate_model.py --bot-model llama3.2:3b            # test a different model
python3 evaluate_model.py --judge-model llama3.1:8b          # use a different judge
python3 evaluate_model.py --test-file eval/my_cases.jsonl    # your own test set
```

**A real limitation worth knowing**: if `--bot-model` and `--judge-model`
are the same model (the default), the judge is grading its own work —
treat scores as a rough signal, not ground truth. If you have a second
or larger model pulled in Ollama, point `--judge-model` at that for a
more independent read. Either way, the report is meant to point you at
transcripts worth reading, not to replace reading them.

You can add your own test cases by appending lines to
`eval/test_cases.jsonl` (or a new file) in the same format:
```json
{"id": "my_test_01", "type": "single_turn", "dimension_focus": ["validity"], "turns": ["Your question here"]}
```
Use `"type": "multi_turn"` with multiple strings in `"turns"` to test
follow-up questions and conversational progress.

## Running tests

```bash
python3 -m pytest
```

## Notes & next steps

- **Adding more philosophers**: add entries to `BOOKS` in
  `download_texts.py`, rerun it, then rerun `build_index.py` (and the
  dataset scripts if you're using the fine-tuning pipeline).
- **Speed**: everything runs on CPU by default. If you have a decent
  GPU, Ollama will automatically use it and responses will be much
  faster.
- **Copyright**: all texts here are public domain. If you want to add
  more recent philosophers, check their copyright status first —
  Gutenberg only hosts public-domain works.
