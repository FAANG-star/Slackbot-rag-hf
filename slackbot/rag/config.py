"""RAG pipeline configuration constants."""

from pathlib import Path

# --- Paths (all on /data volume, under /data/rag/ to avoid conflicts with ml_agent) ---
RAG_ROOT = Path("/data/rag")
DOCS_DIR = RAG_ROOT / "docs"
CHROMA_DIR = RAG_ROOT / "chroma"
OUTPUT_DIR = RAG_ROOT / "output"

# --- LLM (vLLM subprocess) ---
VLLM_MODEL = "Qwen/Qwen3-14B-AWQ"  # AWQ 4-bit, ~8GB on A10G
LLM_CONTEXT_WINDOW = 16384

# --- Embeddings ---
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"  # 110M params, 768-dim

# --- Retrieval ---
TOP_K = 3

# --- ChromaDB ---
CHROMA_COLLECTION = "rag_documents"

# --- System prompt ---
SYSTEM_PROMPT = (
    "/no_think\n"
    "You are a document assistant with three tools.\n\n"
    "**list_documents()** — lists all files in /data/rag/docs/. "
    "Call this first when the user mentions a file, to confirm it exists and get the exact path.\n\n"
    "**search_documents(query)** — full-text search over indexed documents. "
    "Use for questions about document content, concepts, or facts.\n\n"
    "**execute_python(code)** — runs Python in a subprocess and returns stdout. "
    "Use for data analysis, file reading, chart generation, or any computation. "
    "Save charts/outputs to /data/rag/output/. "
    "Pre-installed: pandas, matplotlib, openpyxl, pypdf, python-docx.\n\n"
    "Rules:\n"
    "- When the user mentions a file: call list_documents first, then execute_python with the exact path.\n"
    "- For document questions: call search_documents first, then answer from results.\n"
    "- Never guess file contents or paths — use tools to check.\n"
    "- Be concise. Cite sources from search results.\n"
    "- Do not use emojis in responses."
)
