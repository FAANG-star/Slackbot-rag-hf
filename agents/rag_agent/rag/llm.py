"""Local LLM — vLLM OpenAI-compatible server + LlamaIndex OpenAILike client."""

import subprocess
import sys
import time
import urllib.request

from llama_index.llms.openai_like import OpenAILike

from .config import LLM_MODEL

_PORT = 8000
_BASE_URL = f"http://localhost:{_PORT}/v1"


class LLM:
    def __init__(self, model_name: str = LLM_MODEL):
        self.model_name = model_name
        self.model = self._start()

    def _start(self) -> OpenAILike:
        print("[LLM] Starting vLLM server...", file=sys.stderr, flush=True)
        self._proc = subprocess.Popen(
            [
                "python", "-m", "vllm.entrypoints.openai.api_server",
                "--model", self.model_name,
                "--download-dir", "/data/hf-cache",
                "--gpu-memory-utilization", "0.65",
                "--quantization", "awq_marlin",
                "--max-model-len", "16384",
                "--dtype", "auto",
                "--enforce-eager",
                "--chat-template-content-format", "string",
                "--port", str(_PORT),
            ],
            stdout=sys.stderr,  # keep vLLM logs off the response stdout stream
        )
        self._wait_ready()
        print("[LLM] vLLM server ready.", file=sys.stderr, flush=True)
        return OpenAILike(
            model=self.model_name,
            api_base=_BASE_URL,
            api_key="placeholder",
            max_tokens=4096,
            temperature=0.7,
            top_p=0.8,
            is_chat_model=True,
            context_window=16384,
        )

    def _wait_ready(self, timeout: int = 300, interval: float = 2.0):
        deadline = time.monotonic() + timeout
        url = f"http://localhost:{_PORT}/health"
        while time.monotonic() < deadline:
            try:
                urllib.request.urlopen(url, timeout=2)
                return
            except Exception:
                time.sleep(interval)
        raise RuntimeError(f"vLLM server did not start within {timeout}s")
