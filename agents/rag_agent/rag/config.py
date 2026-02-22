"""RAG pipeline configuration constants."""

from pathlib import Path

# --- Paths (all on /data volume, under /data/rag/ to avoid conflicts with ml_agent) ---
RAG_ROOT = Path("/data/rag")
DOCS_DIR = RAG_ROOT / "docs"
CHROMA_DIR = RAG_ROOT / "chroma"
HISTORY_DIR = RAG_ROOT / "history"
OUTPUT_DIR = RAG_ROOT / "output"
MANIFEST_PATH = CHROMA_DIR / "manifest.json"

# --- LLM backend ---
LLM_BACKEND = "vllm"  # "vllm" or "ollama"
LLM_CONTEXT_WINDOW = 16384

# vLLM backend
VLLM_MODEL = "cognitivecomputations/Qwen3-30B-A3B-AWQ"  # MoE AWQ, ~16GB on A10G, ~3B active params

# Ollama backend
OLLAMA_MODEL = "qwen3:14b"
OLLAMA_PORT = 11434

# --- Embeddings ---
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"  # 110M params, 768-dim

# --- Chunking ---
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64

# --- Embedding ---
EMBED_BATCH_SIZE = 256  # texts per GPU forward pass

# --- Retrieval ---
TOP_K = 3

# --- History ---
MAX_HISTORY_TURNS = 10
MEMORY_TOKEN_LIMIT = 4096

# --- ChromaDB ---
CHROMA_COLLECTION = "rag_documents"
