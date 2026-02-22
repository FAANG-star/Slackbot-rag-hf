"""RAG client — query protocol over persistent sandbox stdin/stdout."""

import json
import threading
import traceback

from .response import RagResponse, parse_response
from .sandbox_manager import SandboxManager


class RagClient:
    """Sends queries to the RAG sandbox and returns structured responses."""

    def __init__(self):
        self._lock = threading.Lock()
        self._sb = SandboxManager()

    def is_running(self) -> bool:
        return self._sb.is_running

    def query(self, message: str, sandbox_name: str, on_status=None) -> RagResponse:
        """Send a query and return the response. Retries once on failure."""
        with self._lock:
            msg = json.dumps({"message": message, "sandbox_name": sandbox_name}) + "\n"
            self._send_with_retry(msg, on_status)
            return parse_response(self._sb.stdout)

    def _send_with_retry(self, msg: str, on_status=None) -> None:
        """Try to send, reset and retry once on failure."""
        for attempt in range(2):
            try:
                self._sb.ensure_running(on_status)
                self._sb.sandbox.stdin.write(msg.encode())
                self._sb.sandbox.stdin.drain()
                return
            except Exception as e:
                print(f"[RAG] send failed (attempt {attempt + 1}): {e}", flush=True)
                traceback.print_exc()
                self._sb.reset()
                if attempt == 1:
                    raise
