"""LLM factory — selects backend (vLLM or Ollama) via config."""

import sys

from ..config import LLM_BACKEND


class LLM:
    """Config-driven LLM loader. Both backends expose an OpenAILike `.model`."""

    def __init__(self):
        print(f"[LLM] Using backend: {LLM_BACKEND}", file=sys.stderr, flush=True)
        if LLM_BACKEND == "vllm":
            from .adapters.vllm_backend import VllmBackend
            self._backend = VllmBackend()
        elif LLM_BACKEND == "ollama":
            from .adapters.ollama_backend import OllamaBackend
            self._backend = OllamaBackend()
        else:
            raise ValueError(f"Unknown LLM_BACKEND: {LLM_BACKEND!r}")
        self.model = self._backend.model

    def start(self):
        """Start the server, warm up, and sleep (vLLM only; no-op for others)."""
        if hasattr(self._backend, "start"):
            self._backend.start()

    def wake_up(self):
        """Wake up after snapshot restore (vLLM only; no-op for others)."""
        if hasattr(self._backend, "wake_up"):
            self._backend.wake_up()

    def terminate(self):
        """Terminate the backend subprocess (vLLM only; no-op for others)."""
        if hasattr(self._backend, "terminate"):
            self._backend.terminate()
