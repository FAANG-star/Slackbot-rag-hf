"""Document scanner — finds new/changed files via manifest comparison."""

import json
from pathlib import Path

import modal


class Scanner:

    def __init__(self, volume: modal.Volume, docs_dir: Path, manifest_path: Path):
        self._volume = volume
        self._docs_dir = docs_dir
        self._manifest_path = manifest_path

    def scan(self, force: bool) -> list[str]:
        self._volume.reload()
        if not self._docs_dir.exists():
            return []

        manifest = self._load_manifest()
        all_files = sorted(p for p in self._docs_dir.iterdir() if p.is_file())
        to_index = [
            p for p in all_files
            if force or manifest.get(p.name) != self._fingerprint(p)
        ]

        print(f"[scan] {len(to_index)} to index, {len(all_files) - len(to_index)} already indexed", flush=True)
        for p in to_index:
            print(f"[scan]   {p.name}", flush=True)
        return [str(p) for p in to_index]

    def empty_result(self) -> str:
        if not self._docs_dir.exists():
            return "No documents found. Share some files first."
        manifest = self._load_manifest()
        has_indexed = any(p.name in manifest for p in self._docs_dir.iterdir() if p.is_file())
        if has_indexed:
            return "All documents are already indexed. Share new files or use `reindex --force` to rebuild from scratch."
        return "No documents found. Share some files first."

    def _fingerprint(self, path: Path) -> str:
        stat = path.stat()
        return f"{stat.st_mtime_ns}:{stat.st_size}"

    def _load_manifest(self) -> dict:
        try:
            return json.loads(self._manifest_path.read_text())
        except Exception:
            return {}
