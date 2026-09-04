"""
config.py

Single source of truth for settings shared across the project. Change
the model or embedding settings here instead of hunting through
query.py, build_index.py, generate_synthetic_dataset.py, and
evaluate_model.py separately.

Every script's own defaults (e.g. generate_synthetic_dataset.py's
--model flag) fall back to these values, so CLI flags still let you
override per-run without editing this file.
"""

# Ollama model used to actually answer questions (the "bot"). Change this
# to any model you've pulled -- e.g. "mistral", "llama3.2:3b", "phi3".
OLLAMA_MODEL = "llama3.1:8b"

# Local embedding model (sentence-transformers) used for retrieval.
# This is separate from OLLAMA_MODEL -- it doesn't generate text, it just
# turns text into vectors for the Chroma similarity search.
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

# Chroma vector store location and collection name -- created by
# build_index.py, read by query.py and evaluate_model.py.
DB_DIR = "chroma_db"
COLLECTION_NAME = "philosophers"

# Retrieval tuning (see query.py's retrieve() for how these are used)
TOP_K = 10                     # how many passages to retrieve per question
CANDIDATE_POOL = 30            # how many nearest matches to pull before diversifying
MAX_PER_AUTHOR = 3             # cap per philosopher so one voice doesn't dominate

# Conversation memory: how many previous Q&A exchanges to keep in history.
# Older exchanges beyond this are dropped from context (oldest first) so
# a long chat doesn't eventually exceed the model's context window.
MAX_HISTORY_EXCHANGES = 8