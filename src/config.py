"""Centralised configuration for the RAG pipeline.

All tuneable parameters live here. Change these values to experiment
with different chunking strategies, models, or retrieval settings.
"""

from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PAPERS_DIR = PROJECT_ROOT / "data" / "papers"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"

# ─── Chunking ─────────────────────────────────────────────────────────
CHUNK_SIZE = 1000          # characters per chunk
CHUNK_OVERLAP = 200        # overlap between consecutive chunks
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

# ─── Embedding ────────────────────────────────────────────────────────
EMBEDDING_MODEL = "nomic-embed-text"   # Ollama embedding model
OLLAMA_BASE_URL = "http://localhost:11434"

# ─── Retrieval ────────────────────────────────────────────────────────
RETRIEVER_K = 4            # number of chunks to retrieve
SEARCH_TYPE = "similarity"

# ─── LLM ──────────────────────────────────────────────────────────────
LLM_MODEL = "llama3.2"
LLM_TEMPERATURE = 0.1     # low temp for factual answers
LLM_BACKEND = "ollama"     # "ollama" or "huggingface"

# ─── Hugging Face (alternative backend — free Inference API) ──────────
HF_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
# Set HF_TOKEN as env var or in .env — never commit tokens
