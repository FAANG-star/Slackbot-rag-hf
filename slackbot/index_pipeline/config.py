"""Index pipeline configuration."""

import modal

hf_secret = modal.Secret.from_name("hf-secret")

# EmbedWorker: TEI base image + parsing/chunking libs
embed_image = (
    modal.Image.from_registry(
        "ghcr.io/huggingface/text-embeddings-inference:86-1.7",
        add_python="3.12",
    )
    .entrypoint([])
    .uv_pip_install(
        "llama-index-core",
        "llama-index-readers-file",
        "pypdf",
        "python-docx",
        "httpx",
    )
    .run_commands("python -c \"import nltk; nltk.download('punkt_tab'); nltk.download('stopwords')\"")
)

# UpsertWorker: just ChromaDB
upsert_image = modal.Image.debian_slim(python_version="3.12").pip_install("chromadb")

N_WORKERS = 8
WORKERS_PER_GPU = 4

CHUNK_SIZE = 1024
CHUNK_OVERLAP = 128

CHROMA_DIR = "/data/rag/chroma"
CHROMA_COLLECTION = "rag_documents"
