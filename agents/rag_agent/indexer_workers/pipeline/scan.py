"""Scan phase — reload volume and return files that need indexing."""

import json
from pathlib import Path
from typing import Callable

import modal


class ScanPhase:
    def __init__(self, volume: modal.Volume, docs_dir: Path):
        self._volume = volume
        self._docs_dir = docs_dir

    def run(self, force: bool, on_status: Callable[[str], None]) -> list[str]:
        on_status("scanning documents...")
        self._volume.reload()
        if not self._docs_dir.exists():
            print("[scan] docs dir does not exist", flush=True)
            return []
        manifest = self._load_manifest()
        result = []
        for p in sorted(self._docs_dir.iterdir()):
            if not p.is_file():
                continue
            if not force:
                stat = p.stat()
                if manifest.get(p.name) == f"{stat.st_mtime_ns}:{stat.st_size}":
                    continue
            result.append(str(p))
        skipped = len(list(self._docs_dir.iterdir())) - len(result)
        print(f"[scan] {len(result)} file(s) to index, {skipped} already indexed", flush=True)
        for f in result:
            print(f"[scan]   {Path(f).name}", flush=True)
        return result

    def already_indexed(self) -> list[str]:
        if not self._docs_dir.exists():
            return []
        manifest = self._load_manifest()
        return [p.name for p in self._docs_dir.iterdir() if p.is_file() and p.name in manifest]

    def _load_manifest(self) -> dict:
        try:
            return json.loads(Path("/data/rag/chroma/manifest.json").read_text())
        except Exception:
            return {}
