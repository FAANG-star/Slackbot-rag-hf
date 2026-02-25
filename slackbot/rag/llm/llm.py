"""vLLM subprocess — start, warmup, sleep/wake for GPU snapshots."""

import socket
import subprocess
import sys
import threading
import time

import requests
from llama_index.llms.openai_like import OpenAILike

from ..config import LLM_CONTEXT_WINDOW, VLLM_MODEL

_PORT = 8000
_BASE_URL = f"http://localhost:{_PORT}/v1"

# NCCL heartbeat noise after snapshot restore
_SUPPRESS = (
    "TCPStore.cpp", "ProcessGroupNCCL.cpp", "HeartbeatMonitor",
    "sendBytes failed", "Failed to check the", "Exception raised from sendBytes",
    "libc10.so", "libtorch_cpu.so", "libtorch_cuda.so", "libstdc++.so", "libc.so",
    "frame #",
)


class LLM:
    """Manages vLLM subprocess with sleep/wake for Modal GPU snapshots.

    Lifecycle:
        start()     — launch server, warmup, sleep (snap=True enter)
        wake_up()   — restore weights to GPU (snap=False enter)
        terminate() — kill subprocess (exit)
    """

    def __init__(self):
        self._proc: subprocess.Popen | None = None
        self.model = OpenAILike(
            model="llm",
            api_base=_BASE_URL,
            api_key="vllm",
            max_tokens=4096,
            temperature=0.7,
            top_p=0.8,
            is_chat_model=True,
            context_window=LLM_CONTEXT_WINDOW,
        )

    def start(self):
        """Start vLLM, warm up, then sleep (offload weights for snapshot)."""
        self._start_server()
        self._warmup()
        self._sleep()

    def wake_up(self):
        """Restore GPU weights after snapshot restore."""
        print("[LLM] Waking up...", file=sys.stderr, flush=True)
        requests.post(f"http://localhost:{_PORT}/wake_up").raise_for_status()
        self._wait_ready()
        print("[LLM] Ready.", file=sys.stderr, flush=True)

    def terminate(self):
        if self._proc is not None:
            self._proc.terminate()

    # -- Internal --

    def _start_server(self):
        cmd = [
            "vllm", "serve", VLLM_MODEL,
            "--served-model-name", "llm",
            "--host", "0.0.0.0",
            "--port", str(_PORT),
            "--download-dir", "/data/hf-cache",
            "--dtype", "auto",
            "--quantization", "awq_marlin",
            "--gpu-memory-utilization", "0.90",
            "--max-model-len", str(LLM_CONTEXT_WINDOW),
            "--enforce-eager",
            "--enable-sleep-mode",
            "--max-num-seqs", "4",
        ]
        print("[LLM] Starting vLLM...", file=sys.stderr, flush=True)
        self._proc = subprocess.Popen(cmd, stdout=sys.stderr, stderr=subprocess.PIPE, text=True)
        threading.Thread(target=self._filter_stderr, daemon=True).start()
        self._wait_ready()
        print("[LLM] Server ready.", file=sys.stderr, flush=True)

    def _warmup(self):
        """Run 3 short inferences to warm up GPU kernels."""
        payload = {"model": "llm", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 16}
        for _ in range(3):
            requests.post(f"{_BASE_URL}/chat/completions", json=payload, timeout=300).raise_for_status()

    def _sleep(self):
        """Offload weights to CPU RAM for snapshot."""
        requests.post(f"http://localhost:{_PORT}/sleep?level=1").raise_for_status()
        print("[LLM] Sleeping (weights offloaded).", file=sys.stderr, flush=True)

    def _wait_ready(self, timeout: int = 300, interval: float = 2.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                socket.create_connection(("localhost", _PORT), timeout=1).close()
                return
            except OSError:
                if self._proc is not None and self._proc.poll() is not None:
                    raise RuntimeError(f"vLLM exited with code {self._proc.returncode}")
                time.sleep(interval)
        raise RuntimeError(f"vLLM did not start within {timeout}s")

    def _filter_stderr(self):
        """Forward vLLM stderr, dropping NCCL noise."""
        for line in self._proc.stderr:
            if line.strip() and not any(p in line for p in _SUPPRESS):
                sys.stderr.write(line)
                sys.stderr.flush()
