"""RAG pipeline configuration constants."""

from pathlib import Path

# --- Paths (all on /data volume, under /data/rag/ to avoid conflicts with ml_agent) ---
RAG_ROOT = Path("/data/rag")
DOCS_DIR = RAG_ROOT / "docs"
CHROMA_DIR = RAG_ROOT / "chroma"
HISTORY_DIR = RAG_ROOT / "history"
MANIFEST_PATH = CHROMA_DIR / "manifest.json"

# --- Models ---
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"  # 335M params, 1024-dim, top MTEB
LLM_MODEL = "Qwen/Qwen2.5-14B-Instruct"  # 4-bit via bitsandbytes, ~8GB on A10G

# --- Chunking ---
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64

# --- Retrieval ---
TOP_K = 6

# --- History ---
MAX_HISTORY_TURNS = 10
MEMORY_TOKEN_LIMIT = 4096

# --- ChromaDB ---
CHROMA_COLLECTION = "rag_documents"
