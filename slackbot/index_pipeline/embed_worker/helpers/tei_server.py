"""TEI subprocess lifecycle."""

import socket
import subprocess
import time


class TeiServer:

    def __init__(self, model: str, port: int, max_batch: int):
        self._model = model
        self._port = port
        self._max_batch = max_batch

    def start(self) -> None:
        self._proc = subprocess.Popen([
            "text-embeddings-router",
            "--model-id", self._model,
            "--port", str(self._port),
            "--max-client-batch-size", str(self._max_batch),
            "--auto-truncate",
            "--json-output",
        ])
        self._wait_ready()

    def _wait_ready(self, timeout: float = 120.0) -> None:
        start = time.monotonic()
        while time.monotonic() - start < timeout:
            try:
                with socket.create_connection(("127.0.0.1", self._port), timeout=1.0):
                    print("TEI server ready", flush=True)
                    return
            except OSError:
                time.sleep(0.5)
        raise TimeoutError(f"TEI did not start within {timeout}s")
