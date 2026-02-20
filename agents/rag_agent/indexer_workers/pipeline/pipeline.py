"""IndexPipeline — composes scan, embed, and finalize phases."""

import threading
from pathlib import Path
from typing import Callable

import modal

from .scan import ScanPhase
from .embed import EmbedPhase
from .finalize import FinalizePhase


class IndexPipeline:
    """Orchestrates parallel GPU indexing and reports progress to Slack."""

    def __init__(self, volume: modal.Volume, docs_dir: Path = Path("/data/rag/docs")):
        from agents.rag_agent.indexer_workers import N_WORKERS

        self._scan = ScanPhase(volume, docs_dir)
        self._embed = EmbedPhase(N_WORKERS)
        self._finalize = FinalizePhase()
        self._lock = threading.Lock()

    def reindex(
        self,
        force: bool = False,
        on_status: Callable[[str], None] | None = None,
        reload_fn: Callable[[], None] | None = None,
    ) -> str:
        """Run the full indexing pipeline. Returns a summary string for Slack."""
        _status = on_status or (lambda s: None)
        with self._lock:
            print("[pipeline] === scan ===", flush=True)
            files = self._scan.run(force, _status)
            if not files:
                if self._scan.already_indexed():
                    return "All documents already indexed. Share new files or run `reindex --force` to rebuild."
                return "No documents found in /data/rag/docs/. Share files via Slack first."

            print("[pipeline] === embed ===", flush=True)
            self._embed.run(files, _status)

            print("[pipeline] === finalize ===", flush=True)
            result = self._finalize.run(force, _status)

            if reload_fn:
                print("[pipeline] === reload ===", flush=True)
                _status("reloading search index...")
                reload_fn()

            print("[pipeline] === done ===", flush=True)
            return result
