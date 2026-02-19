"""Tracks file fingerprints on disk for incremental change detection."""

import json
from pathlib import Path

from ..config import DOCS_DIR, MANIFEST_PATH


class Manifest:
    def __init__(self, path: Path = MANIFEST_PATH):
        self._path = path

    @staticmethod
    def _fingerprint(path: Path) -> str:
        stat = path.stat()
        return f"{stat.st_mtime_ns}:{stat.st_size}"

    def load(self) -> dict[str, str]:
        if self._path.exists():
            try:
                return json.loads(self._path.read_text())
            except (json.JSONDecodeError, ValueError):
                return {}
        return {}

    def save(self, manifest: dict[str, str]):
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(manifest, indent=2))

    def scan(self, docs_dir: Path = DOCS_DIR) -> dict[str, str]:
        """Scan a directory and return {filename: fingerprint}."""
        files = {}
        if docs_dir.exists():
            for path in sorted(docs_dir.iterdir()):
                if path.is_file():
                    files[path.name] = self._fingerprint(path)
        return files

    @staticmethod
    def diff(old: dict, current: dict) -> tuple[list[str], set[str]]:
        """Compare two manifests. Returns (to_add, to_delete)."""
        to_add = []
        to_delete = set()

        for fname, fp in current.items():
            if fname not in old or old[fname] != fp:
                to_add.append(fname)
                if fname in old:
                    to_delete.add(fname)

        for fname in old:
            if fname not in current:
                to_delete.add(fname)

        return to_add, to_delete
