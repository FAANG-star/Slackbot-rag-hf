"""Orchestrates scan → parallel embed → finalize for document indexing."""

import threading
import time
from pathlib import Path
from typing import Callable

import modal

from .config import CHROMA_DIR, N_WORKERS, WORKERS_PER_GPU
from .embed_worker import EmbedWorker, UpsertWorker
from .pipeline.batch_builder import BatchBuilder
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
            upsert = UpsertWorker()

            # 1. Scan for new/changed documents
            status("Looking for new documents...")
            files = self._scanner.scan(force)
            if not files:
                return self._scanner.empty_result()

            # 2. Reset collection if force reindex
            if force:
                upsert.reset.remote()

            # 3. Tell upsert worker how many batches to expect
            batches, doc_count = self._batch_builder.build(files)
            upsert.begin.remote(len(batches))

            # 4. Embed (GPU workers .spawn() chunks to upsert worker)
            self._embed(batches, doc_count, status)
            status("Embedding complete, writing to database...")

            # 5. Poll upsert progress until self-finalize completes
            result = self._await_upsert(upsert, status)

            # 6. Reload RAG agent if callback provided
            if reload_fn:
                status("Finishing up...")
                reload_fn()

            return result

    def _embed(self, batches: list[dict], doc_count: int, status: StatusFn) -> None:
        """Distribute files across GPU workers and report progress."""
        status(f"Embedding {doc_count:,} documents across {len(batches)} workers...")

        worker_results = EmbedWorker().embed.map(
            batches, range(len(batches)), return_exceptions=True
        )

        total_chunks, failures = 0, 0
        for i, result in enumerate(worker_results, 1):
            if isinstance(result, Exception):
                failures += 1
                print(f"[embed] worker failed: {result}", flush=True)
            else:
                total_chunks += result[1]
            status(f"Embedding ({i}/{len(batches)} workers, {total_chunks:,} passages)...")

        if failures:
            status(f"Warning: {failures} of {len(batches)} workers failed.")

    def _await_upsert(self, upsert: UpsertWorker, status: StatusFn) -> str:
        """Poll upsert progress, report at 25% thresholds, return final result."""
        reported: set[int] = set()
        while True:
            drained, expected, result = upsert.progress.remote()
            if result:
                return result
            if expected:
                pct = int(drained / expected * 100)
                for threshold in (25, 50, 75):
                    if pct >= threshold and threshold not in reported:
                        status(f"Writing to database ({pct}%)...")
                        reported.add(threshold)
            time.sleep(10)
