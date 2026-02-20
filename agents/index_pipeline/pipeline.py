"""IndexPipeline — orchestrates scan, embed, and finalize phases."""

import json
import math
import threading
from pathlib import Path
from typing import Callable

import modal

from agents.index_pipeline import EmbedWorker, N_WORKERS, finalize_index
from agents.index_pipeline.config import CHROMA_DIR

_MANIFEST_PATH = Path(CHROMA_DIR) / "manifest.json"

Status = Callable[[str], None]


class IndexPipeline:
    """Orchestrates parallel GPU indexing and reports progress to Slack."""

    def __init__(self, volume: modal.Volume, docs_dir: Path = Path("/data/rag/docs")):
        self._volume = volume
        self._docs_dir = docs_dir
        self._lock = threading.Lock()

    # ── Public API ───────────────────────────────────────────────────────────

    def reindex(
        self,
        force: bool = False,
        on_status: Status | None = None,
        reload_fn: Callable[[], None] | None = None,
    ) -> str:
        """Run the full indexing pipeline. Returns a summary string for Slack."""
        _status = on_status or (lambda s: None)
        with self._lock:
            print("[pipeline] === scan ===", flush=True)
            files = self._scan(force, _status)
            if not files:
                return self._empty_result()

            print("[pipeline] === embed ===", flush=True)
            self._embed(files, _status)

            print("[pipeline] === finalize ===", flush=True)
            result = self._finalize(force, _status)

            if reload_fn:
                print("[pipeline] === reload ===", flush=True)
                _status("reloading search index...")
                reload_fn()

            print("[pipeline] === done ===", flush=True)
            return result

    # ── Pipeline phases ──────────────────────────────────────────────────────

    def _scan(self, force: bool, on_status: Status) -> list[str]:
        on_status("scanning documents...")
        self._volume.reload()
        if not self._docs_dir.exists():
            return []

        manifest = self._load_manifest()
        all_files = sorted(p for p in self._docs_dir.iterdir() if p.is_file())
        to_index = [
            p for p in all_files
            if force or manifest.get(p.name) != self._fingerprint(p)
        ]

        print(f"[scan] {len(to_index)} file(s) to index, {len(all_files) - len(to_index)} already indexed", flush=True)
        for p in to_index:
            print(f"[scan]   {p.name}", flush=True)
        return [str(p) for p in to_index]

    def _embed(self, files: list[str], on_status: Status) -> None:
        on_status(f"embedding {len(files)} file(s) across GPU workers...")
        batches, worker_ids = self._make_batches(files)
        print(f"[embed] spawning {len(batches)} workers (~{math.ceil(len(files) / N_WORKERS)} file(s) each)", flush=True)

        total = 0
        for result in EmbedWorker().embed.map(batches, worker_ids, return_exceptions=True):
            if isinstance(result, Exception):
                print(f"[embed] worker failed: {result}", flush=True)
            else:
                wid, count = result
                total += count
                on_status(f"worker-{wid} done: {count:,} chunks (total: {total:,})")
        print(f"[embed] all workers done: {total:,} chunks", flush=True)

    def _make_batches(self, files: list[str]) -> tuple[list[list[str]], list[int]]:
        chunk_size = math.ceil(len(files) / N_WORKERS)
        batches = [files[i : i + chunk_size] for i in range(0, len(files), chunk_size)]
        return batches, list(range(len(batches)))

    def _finalize(self, force: bool, on_status: Status) -> str:
        on_status("writing to ChromaDB...")
        print("[finalize] merging worker manifests...", flush=True)
        result = finalize_index.remote(force)
        print(f"[finalize] {result}", flush=True)
        return result

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _empty_result(self) -> str:
        if not self._docs_dir.exists():
            return "No documents found in /data/rag/docs/. Share files via Slack first."
        manifest = self._load_manifest()
        has_indexed = any(p.name in manifest for p in self._docs_dir.iterdir() if p.is_file())
        if has_indexed:
            return "All documents already indexed. Share new files or run `reindex --force` to rebuild."
        return "No documents found in /data/rag/docs/. Share files via Slack first."

    def _fingerprint(self, path: Path) -> str:
        stat = path.stat()
        return f"{stat.st_mtime_ns}:{stat.st_size}"

    def _load_manifest(self) -> dict:
        try:
            return json.loads(_MANIFEST_PATH.read_text())
        except Exception:
            return {}
