"""Orchestrates scan → parallel embed → finalize for document indexing."""

import threading
from pathlib import Path
from typing import Callable

from .config import CHROMA_DIR, N_WORKERS, WORKERS_PER_GPU
from .embed_worker import EmbedWorker
from .upsert_worker import UpsertWorker
from .pipeline.batch_builder import BatchBuilder
from .pipeline.scanner import Scanner

StatusFn = Callable[[str], None]


class IndexPipeline:
    """Scans for new docs, distributes embedding across GPU workers, finalizes the index."""

    def __init__(self, docs_dir: Path = Path("/data/rag/docs")):
        manifest_path = Path(CHROMA_DIR) / "manifest.json"
        self._scanner = Scanner(docs_dir, manifest_path)
        self._batch_builder = BatchBuilder(N_WORKERS * WORKERS_PER_GPU)
        self._lock = threading.Lock()

    def reindex(
        self,
        force: bool = False,
        on_status: StatusFn | None = None,
        slack: dict | None = None,
    ) -> str | None:
        """Run the full indexing pipeline. Returns early result or None (async)."""
        status = on_status or (lambda _: None)

        with self._lock:
            # 1. Scan
            status("Looking for new documents...")
            files = self._scanner.scan(force)
            if not files:
                return self._scanner.empty_result()

            # 2. Reset if force
            upsert = UpsertWorker()
            if force:
                upsert.reset.remote()

            # 3. Begin + spawn (fire-and-forget)
            batches, doc_count = self._batch_builder.build(files)
            upsert.begin.remote(len(batches), slack=slack)
            status(f"Embedding {doc_count:,} documents across {len(batches)} workers...")
            for i, batch in enumerate(batches):
                EmbedWorker().embed.spawn(batch, i)

            return None
