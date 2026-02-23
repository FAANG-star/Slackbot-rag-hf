"""RAG agent image definition — GPU image with vLLM + LlamaIndex, models cached on volume."""

from pathlib import Path

import modal

from agents.slackbot.infra import app, rag_vol

RAG_AGENT_DIR = Path(__file__).parent

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install(
        "vllm==0.13.0",
        "requests",
        "llama-index-core",
        "llama-index-llms-openai-like",
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
    .run_commands("python -c 'from llama_index.llms.openai_like import OpenAILike; from chromadb import PersistentClient; print(\"OK\")'")
    .env({"IMAGE_VERSION": "92", "TORCHINDUCTOR_COMPILE_THREADS": "1", "VLLM_SERVER_DEV_MODE": "1", "TORCH_NCCL_ENABLE_MONITORING": "0"})  # bump BEFORE add_local_dir to invalidate code layers
    # Models download to volume at runtime (cached across restarts)
    .add_local_dir(str(RAG_AGENT_DIR / "server"), "/agent/server", copy=True)
    .add_local_dir(str(RAG_AGENT_DIR / "rag"), "/agent/rag", copy=True)
)


@app.function(image=image)
def _prebuild():
    """Forces image build during `modal deploy`."""
    pass
