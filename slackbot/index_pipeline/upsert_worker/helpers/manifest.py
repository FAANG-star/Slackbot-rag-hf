"""Manifest file I/O — tracks which files have been indexed."""

import json
from pathlib import Path

import modal

_MAIN = "manifest.json"
_WORKER_GLOB = "manifest-worker-*.json"


class Manifest:

    def __init__(self, chroma_dir: str, volume: modal.Volume):
        self._dir = Path(chroma_dir)
        self._vol = volume

    def write_worker(self, worker_id: int, work: dict) -> None:
        """Write a per-worker manifest with file fingerprints, then commit."""
        entries = {}
        for p in self._file_paths(work):
            stat = p.stat()
            entries[p.name] = f"{stat.st_mtime_ns}:{stat.st_size}"
        self._write(f"manifest-worker-{worker_id}.json", entries)

    def merge(self) -> dict:
        """Merge all worker manifests into the main manifest. Returns merged dict."""
        manifest = self._read(_MAIN)
        for wf in sorted(self._dir.glob(_WORKER_GLOB)):
            manifest.update(json.loads(wf.read_text()))
            wf.unlink()
        self._write(_MAIN, manifest)
        return manifest

    def clear(self) -> None:
        for f in self._dir.glob("manifest*.json"):
            f.unlink()

    def _file_paths(self, work: dict) -> list[Path]:
        raw = work.get("paths", [work["zip_path"]] if "zip_path" in work else [])
        return [Path(p) for p in raw]

    def _read(self, name: str) -> dict:
        try:
            return json.loads((self._dir / name).read_text())
        except FileNotFoundError:
            return {}

    def _write(self, name: str, data: dict) -> None:
        (self._dir / name).write_text(json.dumps(data, indent=2))
        self._vol.commit()
