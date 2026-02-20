"""Shared config for indexer workers — image, secrets, constants."""

import modal

from agents.slackbot.shared import app, rag_vol

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
)

EMBED_BATCH_SIZE = 1024
UPSERT_BATCH = 5_000
N_WORKERS = 10
WORKERS_PER_GPU = 4
