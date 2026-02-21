"""Shared config for indexer workers — image, secrets, constants."""

import modal

from agents.slackbot.infra import app, rag_vol

hf_secret = modal.Secret.from_name("hf-secret")

index_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install(
        "llama-index-core",
        "llama-index-embeddings-huggingface",
        "llama-index-readers-file",
        "sentence-transformers>=3.0",
        "transformers>=4.51",
        "chromadb",
        "pypdf",
        "python-docx",
    )
    .run_commands("python -c \"import nltk; nltk.download('punkt_tab'); nltk.download('stopwords')\"")
)

# --- Parallelism ---
N_WORKERS = 10
WORKERS_PER_GPU = 1

# --- Embedding ---
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
EMBED_BATCH_SIZE = 1024   # documents per zip-streaming batch
MODEL_BATCH_SIZE = 64     # chunks per GPU forward pass (fits T4 with WORKERS_PER_GPU=4)

# --- Chunking ---
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64

# --- ChromaDB ---
CHROMA_DIR = "/data/rag/chroma"
CHROMA_COLLECTION = "rag_documents"
UPSERT_BATCH = 5_000
