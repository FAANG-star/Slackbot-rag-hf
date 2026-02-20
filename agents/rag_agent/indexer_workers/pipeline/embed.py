"""Embed phase — fan out embedding work to parallel GPU workers."""

import math
from typing import Callable


class EmbedPhase:
    def __init__(self, n_workers: int):
        self._n_workers = n_workers

    def run(self, files: list[str], on_status: Callable[[str], None]) -> int:
        from agents.rag_agent.indexer_workers import EmbedWorker

        on_status(f"embedding {len(files)} file(s) across GPU workers...")
        chunk_size = math.ceil(len(files) / self._n_workers)
        batches = [files[i : i + chunk_size] for i in range(0, len(files), chunk_size)]
        worker_ids = list(range(len(batches)))
        print(f"[indexer] spawning {len(batches)} workers (~{chunk_size} file(s) each)", flush=True)

        total = 0
        for result in EmbedWorker().embed.map(batches, worker_ids, return_exceptions=True):
            if isinstance(result, Exception):
                print(f"[indexer] worker failed: {result}", flush=True)
            else:
                wid, count = result
                total += count
                on_status(f"worker-{wid} done: {count:,} chunks (total: {total:,})")

        print(f"[indexer] all workers done: {total:,} chunks", flush=True)
        return total
