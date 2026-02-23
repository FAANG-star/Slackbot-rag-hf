"""Ollama backend — GPU inference via Ollama's OpenAI-compatible API."""

import json
import subprocess
import sys
import time
import urllib.request

from llama_index.llms.openai_like import OpenAILike

from ...config import LLM_CONTEXT_WINDOW, OLLAMA_MODEL, OLLAMA_PORT

_BASE_URL = f"http://localhost:{OLLAMA_PORT}/v1"


class OllamaBackend:
    def __init__(self):
        self._start_server()
        self._ensure_model()
        self._warmup()
        self.model = self._build_model()

    def _start_server(self):
        print("[LLM] Starting Ollama server...", file=sys.stderr, flush=True)
        self._proc = subprocess.Popen(
            ["ollama", "serve"],
            stdout=sys.stderr,
            stderr=sys.stderr,
        )
        self._wait_ready()
        print("[LLM] Ollama server ready.", file=sys.stderr, flush=True)

    def _ensure_model(self):
        """Pull model if not already cached on volume."""
        result = subprocess.run(
            ["ollama", "list"], capture_output=True, text=True
        )
        tag = OLLAMA_MODEL if ":" in OLLAMA_MODEL else f"{OLLAMA_MODEL}:latest"
        if tag in result.stdout:
            print(f"[LLM] Model cached: {OLLAMA_MODEL}", file=sys.stderr, flush=True)
            return
        print(f"[LLM] Pulling {OLLAMA_MODEL}...", file=sys.stderr, flush=True)
        subprocess.run(["ollama", "pull", OLLAMA_MODEL], check=True)
        print(f"[LLM] Pull complete: {OLLAMA_MODEL}", file=sys.stderr, flush=True)

    def _warmup(self):
        """Force model load into GPU by sending a short request."""
        print("[LLM] Warming up model (loading to GPU)...", file=sys.stderr, flush=True)
        body = json.dumps({
            "model": OLLAMA_MODEL,
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 1,
        }).encode()
        req = urllib.request.Request(
            f"http://localhost:{OLLAMA_PORT}/v1/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=300)
        print("[LLM] Model loaded to GPU.", file=sys.stderr, flush=True)

    def _build_model(self) -> OpenAILike:
        return OpenAILike(
            model=OLLAMA_MODEL,
            api_base=_BASE_URL,
            api_key="ollama",
            max_tokens=4096,
            temperature=0.7,
            top_p=0.8,
            is_chat_model=True,
            context_window=LLM_CONTEXT_WINDOW,
        )

    def _wait_ready(self, timeout: int = 60, interval: float = 1.0):
        deadline = time.monotonic() + timeout
        url = f"http://localhost:{OLLAMA_PORT}/api/tags"
        while time.monotonic() < deadline:
            try:
                urllib.request.urlopen(url, timeout=2)
                return
            except Exception:
                time.sleep(interval)
        raise RuntimeError(f"Ollama server did not start within {timeout}s")
