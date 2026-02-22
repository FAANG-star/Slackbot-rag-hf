"""Index pipeline configuration."""

import modal

hf_secret = modal.Secret.from_name("hf-secret")

index_image = (
    modal.Image.from_registry(
        "ghcr.io/huggingface/text-embeddings-inference:86-1.7",
        add_python="3.12",
    )
    .entrypoint([])
    .uv_pip_install(
        "llama-index-core",
        "llama-index-readers-file",
        "chromadb",
        "pypdf",
        "python-docx",
        "httpx",
    )
    .run_commands("python -c \"import nltk; nltk.download('punkt_tab'); nltk.download('stopwords')\"")
)

N_WORKERS = 8
WORKERS_PER_GPU = 4

EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
TEI_PORT = 8000
TEI_MAX_BATCH = 512
TEI_BATCH_SIZE = 256

EMBED_BATCH_SIZE = 1024
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64

CHROMA_DIR = "/data/rag/chroma"
CHROMA_COLLECTION = "rag_documents"
UPSERT_BATCH = 5_000
