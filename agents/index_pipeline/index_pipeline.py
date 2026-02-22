"""Orchestrates scan → parallel embed → finalize for document indexing."""

import threading
from pathlib import Path
from typing import Callable

import modal

from .config import CHROMA_DIR, N_WORKERS, WORKERS_PER_GPU
from .embed_worker import EmbedWorker
from .pipeline.batch_builder import BatchBuilder
from .pipeline.finalize import finalize_index
from .pipeline.scanner import Scanner

StatusFn = Callable[[str], None]


class IndexPipeline:
    """Scans for new docs, distributes embedding across GPU workers, finalizes the index."""

    def __init__(self, volume: modal.Volume, docs_dir: Path = Path("/data/rag/docs")):
        manifest_path = Path(CHROMA_DIR) / "manifest.json"
        self._scanner = Scanner(volume, docs_dir, manifest_path)
        self._batch_builder = BatchBuilder(N_WORKERS * WORKERS_PER_GPU)
        self._lock = threading.Lock()

    def reindex(
        self,
        force: bool = False,
        on_status: StatusFn | None = None,
        reload_fn: Callable[[], None] | None = None,
    ) -> str:
        """Run the full indexing pipeline. Thread-safe via lock."""
        status = on_status or (lambda _: None)

        with self._lock:
            # 1. Scan for new/changed documents
            status("Looking for new documents...")
            files = self._scanner.scan(force)
            if not files:
                return self._scanner.empty_result()

            # 2. Distribute across GPU workers
            self._embed(files, status)

            # 3. Merge manifests and build final index
            status("Building search index...")
            result = finalize_index.remote(force)

            # 4. Reload RAG agent if callback provided
            if reload_fn:
                status("Finishing up...")
                reload_fn()

            return result

    def _embed(self, files: list[str], status: StatusFn) -> None:
        """Distribute files across GPU workers and report progress."""
        batches, doc_count = self._batch_builder.build(files)
        status(f"Processing {doc_count:,} documents...")

        # Fan out to GPU workers
        worker_results = EmbedWorker().embed.map(
            batches, range(len(batches)), return_exceptions=True
        )

        # Collect results and report progress
        total_chunks, failures = 0, 0
        for i, result in enumerate(worker_results, 1):
            if isinstance(result, Exception):
                failures += 1
                print(f"[embed] worker failed: {result}", flush=True)
            else:
                total_chunks += result[1]
            status(f"Processing ({i}/{len(batches)} workers, {total_chunks:,} passages)...")

        if failures:
            status(f"Warning: {failures} of {len(batches)} workers failed.")
