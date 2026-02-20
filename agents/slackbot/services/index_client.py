"""Orchestrates parallel GPU indexing and reports progress to Slack."""

import math
import threading
from pathlib import Path
from typing import Callable

import modal


class IndexClient:
    """Fan-out embedding to GPU workers, finalize into ChromaDB, trigger reload."""

    def __init__(self, volume: modal.Volume, docs_dir: Path = Path("/data/rag/docs")):
        self._volume = volume
        self._docs_dir = docs_dir
        self._lock = threading.Lock()

    def reindex(
        self,
        force: bool = False,
        on_status: Callable[[str], None] | None = None,
        reload_fn: Callable[[], None] | None = None,
    ) -> str:
        """Full parallel reindex. Returns summary text for Slack."""
        with self._lock:
            return self._run(force, on_status or (lambda s: None), reload_fn)

    def _run(
        self,
        force: bool,
        on_status: Callable[[str], None],
        reload_fn: Callable[[], None] | None,
    ) -> str:
        from agents.rag_agent.indexer_workers import (
            EmbedWorker,
            N_WORKERS,
            finalize_index,
        )

        on_status("scanning documents...")
        self._volume.reload()
        all_files = self._list_docs()
        if not all_files:
            return "No documents found in /data/rag/docs/. Share files via Slack first."

        on_status(f"embedding {len(all_files)} file(s) across GPU workers...")
        error = self._fan_out(all_files, EmbedWorker().embed, N_WORKERS, on_status)
        if error:
            return error

        on_status("writing to ChromaDB...")
        result = finalize_index.remote(force)

        if reload_fn:
            on_status("reloading search index...")
            reload_fn()

        return result

    def _fan_out(self, all_files, embed_method, n_workers, on_status) -> str | None:
        """Map work to GPU workers. Returns error string or None on success."""
        chunk_size = math.ceil(len(all_files) / n_workers)
        batches = [all_files[i : i + chunk_size] for i in range(0, len(all_files), chunk_size)]
        worker_ids = list(range(len(batches)))
        print(f"[indexer] spawning {len(batches)} workers (~{chunk_size} file(s) each)", flush=True)

        total_chunks = 0
        failed = 0
        for result in embed_method.map(batches, worker_ids, return_exceptions=True):
            if isinstance(result, Exception):
                failed += 1
                print(f"[indexer] worker failed: {result}", flush=True)
            else:
                wid, count = result
                total_chunks += count
                on_status(f"worker-{wid} done: {count:,} chunks (total: {total_chunks:,})")

        if failed:
            return (
                f"Embedding failed for {failed} worker(s). "
                f"{total_chunks:,} chunks succeeded. "
                f"Run `reindex` again to retry."
            )
        print(f"[indexer] all workers done: {total_chunks:,} chunks", flush=True)
        return None

    def _list_docs(self) -> list[str]:
        if not self._docs_dir.exists():
            return []
        return [str(p) for p in sorted(self._docs_dir.iterdir()) if p.is_file()]
