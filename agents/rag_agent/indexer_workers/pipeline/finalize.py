"""Finalize phase — merge worker manifests and report summary."""

from typing import Callable


class FinalizePhase:
    def run(self, force: bool, on_status: Callable[[str], None]) -> str:
        from agents.rag_agent.indexer_workers import finalize_index

        on_status("writing to ChromaDB...")
        print("[finalize] merging worker manifests...", flush=True)
        result = finalize_index.remote(force)
        print(f"[finalize] {result}", flush=True)
        return result
