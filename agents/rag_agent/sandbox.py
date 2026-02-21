"""RAG agent sandbox — local LLM + ChromaDB on GPU, models cached on volume."""

from pathlib import Path

import modal

from agents.slackbot.infra import app, rag_vol

RAG_AGENT_DIR = Path(__file__).parent

SANDBOX_NAME = "rag-agent"

sandbox_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install(
        "vllm==0.13.0",
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
    .run_commands("python -c 'import vllm.entrypoints.openai.api_server; print(\"vllm OK\")'")
    .run_commands("python -c 'from llama_index.llms.openai_like import OpenAILike; from llama_index.core.agent.workflow import AgentWorkflow, ReActAgent; from chromadb import PersistentClient; print(\"llama-index OK\")'")
    .env({"IMAGE_VERSION": "63"})  # bump BEFORE add_local_dir to invalidate code layers
    # Models download to volume at runtime (cached across restarts)
    .add_local_dir(str(RAG_AGENT_DIR / "server"), "/agent/server", copy=True)
    .add_local_dir(str(RAG_AGENT_DIR / "rag"), "/agent/rag", copy=True)
)


@app.function(image=sandbox_image)
def _prebuild():
    """Forces sandbox image build during `modal deploy`."""
    pass
