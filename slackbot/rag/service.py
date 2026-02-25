"""RAG agent — GPU image, vLLM + LlamaIndex, memory snapshot for fast cold starts.

Models are cached on the rag volume. First deploy downloads them (~4 min),
subsequent cold starts restore from GPU snapshot (~1s).
"""

import modal

from slackbot.modal_app import app, rag_vol

# -- GPU image: CUDA + vLLM + LlamaIndex + doc parsing libs --

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
        "transformers>=4.51",
        "sentence-transformers>=3.0",
        "pypdf",
        "python-docx",
        "pandas",
        "openpyxl",
        "matplotlib",
    )
    .env({
        "IMAGE_VERSION": "95",
        "TORCHINDUCTOR_COMPILE_THREADS": "1",
        # Dev mode enables /sleep and /wake_up endpoints for GPU snapshots
        "VLLM_SERVER_DEV_MODE": "1",
        # Suppress NCCL heartbeat noise after snapshot restore
        "TORCH_NCCL_ENABLE_MONITORING": "0",
    })
)


@app.cls(
    image=image,
    gpu="A10G",
    volumes={"/data": rag_vol},
    scaledown_window=60 * 2,
    min_containers=0,
    # GPU snapshot: vLLM sleeps weights to CPU at snap=True, restores at snap=False
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
    startup_timeout=600,
)
@modal.concurrent(max_inputs=5)
class RagService:
    """Local LLM RAG agent with GPU memory snapshot lifecycle."""

    # -- Lifecycle: load → snapshot → wake_up → serve/query → stop --

    @modal.enter(snap=True)
    def load(self):
        """Load LLM and search index, then start vLLM (sleeps for snapshot automatically)."""
        from slackbot.rag.db import SearchIndex
        from slackbot.rag.llm import LLM

        self._llm = LLM()
        self._search_index = SearchIndex()
        self._llm.start()  # starts vLLM subprocess, warms up, then sleeps weights to CPU

    @modal.enter(snap=False)
    def wake_up(self):
        """After snapshot restore: reconnect ChromaDB (stale from snapshot), wake GPU."""
        self._search_index.reload()
        self._llm.wake_up()

    @modal.exit()
    def stop(self):
        self._llm.terminate()

    # -- Interface --

    @modal.web_server(port=8000)
    def serve(self):
        """Expose vLLM's OpenAI-compatible API (subprocess already listening)."""
        pass

    @modal.method()
    def query(self, message: str) -> tuple[str, list[str]]:
        """Run a RAG query. Returns (response_text, output_file_paths)."""
        import asyncio
        from slackbot.rag.agent import parse_response, run_query

        response = asyncio.run(run_query(message, llm=self._llm, search_index=self._search_index))
        return parse_response(response)
