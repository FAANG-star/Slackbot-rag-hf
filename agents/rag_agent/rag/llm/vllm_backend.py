"""vLLM backend — subprocess inference with sleep/wake for GPU snapshots."""

import socket
import subprocess
import sys
import threading
import time

import requests
from llama_index.llms.openai_like import OpenAILike

from ..config import LLM_CONTEXT_WINDOW, VLLM_MODEL

VLLM_PORT = 8000
_BASE_URL = f"http://localhost:{VLLM_PORT}/v1"

# Patterns in vLLM stderr to suppress (NCCL heartbeat noise after snapshot restore)
_SUPPRESS = (
    "TCPStore.cpp",
    "ProcessGroupNCCL.cpp",
    "HeartbeatMonitor",
    "sendBytes failed",
    "Failed to check the",
    "Exception raised from sendBytes",
    "libc10.so",
    "libtorch_cpu.so",
    "libtorch_cuda.so",
    "libstdc++.so",
    "libc.so",
    "frame #",
)


def _pipe_stderr(src):
    """Forward vLLM stderr to sys.stderr, dropping NCCL heartbeat noise lines."""
    for line in src:
        if line.strip() and not any(p in line for p in _SUPPRESS):
            sys.stderr.write(line)
            sys.stderr.flush()


class VllmBackend:
    """Runs vLLM as a subprocess with --enable-sleep-mode for GPU snapshots.

    Usage:
        backend = VllmBackend()
        backend.start()   # call in snap=True enter (starts, warms up, then sleeps)
        backend.wake_up() # call in snap=False enter (restores weights to GPU)
    """

    def __init__(self):
        self._proc: subprocess.Popen | None = None
        self.model = self._build_model()

    def start(self):
        """Start vLLM server, warm up, then sleep (offload weights to CPU for snapshot)."""
        self._start_server()
        self._warmup()
        self._sleep()

    def wake_up(self):
        """Restore GPU weights after snapshot restore."""
        print("[LLM] Waking up vLLM (restoring weights to GPU)...", file=sys.stderr, flush=True)
        requests.post(f"http://localhost:{VLLM_PORT}/wake_up").raise_for_status()
        self._wait_ready()
        print("[LLM] vLLM ready.", file=sys.stderr, flush=True)

    def terminate(self):
        if self._proc is not None:
            self._proc.terminate()

    def _start_server(self):
        print("[LLM] Starting vLLM server...", file=sys.stderr, flush=True)
        cmd = [
            "vllm", "serve",
            VLLM_MODEL,
            "--served-model-name", "llm",
            "--host", "0.0.0.0",
            "--port", str(VLLM_PORT),
            "--download-dir", "/data/hf-cache",
            "--dtype", "auto",
            "--quantization", "awq_marlin",
            "--gpu-memory-utilization", "0.85",
            "--max-model-len", str(LLM_CONTEXT_WINDOW),
            "--enforce-eager",
            "--enable-sleep-mode",
            "--max-num-seqs", "4",
        ]
        print(*cmd, file=sys.stderr, flush=True)
        self._proc = subprocess.Popen(cmd, stdout=sys.stderr, stderr=subprocess.PIPE, text=True)
        threading.Thread(target=_pipe_stderr, args=(self._proc.stderr,), daemon=True).start()
        self._wait_ready()
        print("[LLM] vLLM server ready.", file=sys.stderr, flush=True)

    def _warmup(self):
        print("[LLM] Running warmup inference...", file=sys.stderr, flush=True)
        payload = {
            "model": "llm",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 16,
        }
        for _ in range(3):
            requests.post(
                f"{_BASE_URL}/chat/completions",
                json=payload,
                timeout=300,
            ).raise_for_status()
        print("[LLM] Warmup complete.", file=sys.stderr, flush=True)

    def _sleep(self):
        print("[LLM] Sleeping vLLM (offloading weights to CPU for snapshot)...", file=sys.stderr, flush=True)
        requests.post(f"http://localhost:{VLLM_PORT}/sleep?level=1").raise_for_status()
        print("[LLM] vLLM sleeping — snapshot can be taken.", file=sys.stderr, flush=True)

    def _wait_ready(self, timeout: int = 300, interval: float = 2.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                socket.create_connection(("localhost", VLLM_PORT), timeout=1).close()
                return
            except OSError:
                if self._proc is not None and self._proc.poll() is not None:
                    raise RuntimeError(f"vLLM exited with code {self._proc.returncode}")
                time.sleep(interval)
        raise RuntimeError(f"vLLM did not start within {timeout}s")

    def _build_model(self) -> OpenAILike:
        return OpenAILike(
            model="llm",
            api_base=_BASE_URL,
            api_key="vllm",
            max_tokens=4096,
            temperature=0.7,
            top_p=0.8,
            is_chat_model=True,
            context_window=LLM_CONTEXT_WINDOW,
        )
