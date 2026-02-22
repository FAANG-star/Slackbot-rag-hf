"""RAG client — direct Modal method calls to RagService."""

import threading

from agents.rag_agent.app import RagService

from .config import RagResponse


class RagClient:
    """Calls RagService Modal methods and returns structured responses."""

    def __init__(self):
        self._lock = threading.Lock()
        self._svc = RagService()

    def is_running(self) -> bool:
        # RagService has min_containers=1 — always warm.
        return True

    def query(self, message: str, session_id: str, on_status=None) -> RagResponse:
        """Send a query and return the response."""
        with self._lock:
            text, output_files = self._svc.query.remote(message, session_id)
            return RagResponse(text=text, output_files=output_files)

    def command(self, cmd: str) -> str:
        """Send a command (reload/reindex/status/clear) and return output."""
        return self._svc.command.remote(cmd)
