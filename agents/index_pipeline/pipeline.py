"""IndexPipeline — orchestrates scan, embed, and finalize phases."""

import json
import math
import threading
from pathlib import Path
from typing import Callable

import modal

from agents.index_pipeline import EmbedWorker, N_WORKERS, finalize_index
from agents.index_pipeline.config import WORKERS_PER_GPU
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
            doc_count = self._embed(files, _status)

            print("[pipeline] === finalize ===", flush=True)
            result = self._finalize(force, doc_count, _status)

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

    def _embed(self, files: list[str], on_status: Status) -> int:
        """Run embedding workers. Returns total document count."""
        batches, doc_count = self._build_batches(files)
        worker_ids = list(range(len(batches)))
        on_status(f"embedding {doc_count:,} documents across {len(batches)} GPU workers...")
        print(f"[embed] spawning {len(batches)} workers for {doc_count:,} documents", flush=True)

        total = 0
        for result in EmbedWorker().embed.map(batches, worker_ids, return_exceptions=True):
            if isinstance(result, Exception):
                print(f"[embed] worker failed: {result}", flush=True)
            else:
                wid, count = result
                total += count
                on_status(f"worker-{wid} done: {count:,} chunks (total: {total:,})")
        print(f"[embed] all workers done: {total:,} chunks", flush=True)
        return doc_count

    def _build_batches(self, files: list[str]) -> tuple[list[dict], int]:
        """Split files into per-worker work items. Returns (batches, total_doc_count).

        Zip files are expanded: their entries are enumerated and distributed
        across all GPU slots so each worker gets a slice of entries rather than
        the whole zip. Regular files are distributed by file count.
        """
        import zipfile

        n_batches = N_WORKERS * WORKERS_PER_GPU
        batches: list[dict] = []
        doc_count = 0
        regular = [f for f in files if not f.endswith(".zip")]
        zips = [f for f in files if f.endswith(".zip")]

        # Regular files: split across workers
        if regular:
            doc_count += len(regular)
            chunk_size = math.ceil(len(regular) / n_batches)
            for i in range(0, len(regular), chunk_size):
                batches.append({"type": "files", "paths": regular[i : i + chunk_size]})

        # Zip files: enumerate entries, split entries across workers
        for zip_path in zips:
            with zipfile.ZipFile(zip_path) as zf:
                entries = [n for n in zf.namelist() if not n.endswith("/")]
            doc_count += len(entries)
            print(f"[embed] {zip_path.split('/')[-1]}: {len(entries):,} entries → {n_batches} batches across {N_WORKERS} GPUs", flush=True)
            chunk_size = math.ceil(len(entries) / n_batches)
            for i in range(0, len(entries), chunk_size):
                batches.append({
                    "type": "zip_entries",
                    "zip_path": zip_path,
                    "entries": entries[i : i + chunk_size],
                })

        return batches, doc_count

    def _finalize(self, force: bool, doc_count: int, on_status: Status) -> str:
        on_status("finalizing index...")
        print("[finalize] merging worker manifests...", flush=True)
        result = finalize_index.remote(force, doc_count)
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
