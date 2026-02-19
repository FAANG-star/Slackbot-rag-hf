"""RAG agent sandbox — local LLM + ChromaDB on GPU, models cached on volume."""

from pathlib import Path

import modal

from agents.infra.shared import app, rag_vol

RAG_AGENT_DIR = Path(__file__).parent

SANDBOX_NAME = "rag-agent"

sandbox_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "curl")
    .pip_install("torch")
    .pip_install(
        "llama-index-core",
        "llama-index-vector-stores-chroma",
        "llama-index-embeddings-huggingface",
        "llama-index-llms-huggingface",
        "llama-index-readers-file",
        "bitsandbytes",
        "transformers>=4.44", "accelerate", "sentence-transformers>=3.0",
        "pypdf", "python-docx",
        "pandas", "openpyxl", "matplotlib",
    )
    # Models download to volume at runtime (cached across restarts)
    .add_local_dir(str(RAG_AGENT_DIR), "/agent", copy=True)
    .env({"IMAGE_VERSION": "21"})  # bump to force image rebuild
)


@app.function(image=sandbox_image)
def _prebuild():
    """Forces sandbox image build during `modal deploy`."""
    pass


def get_sandbox() -> modal.Sandbox:
    """Get the existing RAG sandbox or create a new one."""
    try:
        return modal.Sandbox.from_name(app_name=app.name, name=SANDBOX_NAME)
    except modal.exception.NotFoundError:
        return modal.Sandbox.create(
            app=app,
            image=sandbox_image,
            workdir="/agent",
            volumes={"/data": rag_vol},
            gpu="A10G",
            env={"HF_HOME": "/data/hf-cache"},
            timeout=60 * 60,
            name=SANDBOX_NAME,
            idle_timeout=20 * 60,
        )
