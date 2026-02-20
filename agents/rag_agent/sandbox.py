"""RAG agent sandbox — local LLM + ChromaDB on GPU, models cached on volume."""

from pathlib import Path

import modal

from agents.infra.shared import app, rag_vol

RAG_AGENT_DIR = Path(__file__).parent

SANDBOX_NAME = "rag-agent"

sandbox_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install(
        "vllm==0.13.0",
        "llama-index-core",
        "llama-index-vector-stores-chroma",
        "llama-index-embeddings-huggingface",
        "llama-index-readers-file",
        "transformers>=4.51",
        "sentence-transformers>=3.0",
        "pypdf",
        "python-docx",
        "pandas",
        "openpyxl",
        "matplotlib",
    )
    .run_commands("python -c 'from vllm import LLM, SamplingParams; print(\"vllm OK\")'")
    .run_commands("python -c 'from llama_index.core.agent.workflow import AgentWorkflow, ReActAgent; from chromadb import PersistentClient; print(\"llama-index OK\")'")
    # Models download to volume at runtime (cached across restarts)
    .add_local_dir(str(RAG_AGENT_DIR), "/agent", copy=True)
    .env({"IMAGE_VERSION": "40"})  # bump to force image rebuild
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
            "sleep", "infinity",
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
