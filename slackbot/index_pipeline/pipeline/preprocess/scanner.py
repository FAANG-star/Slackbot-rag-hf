"""Document scanner — finds new/changed files via ChromaDB comparison."""

from pathlib import Path
from typing import Callable

from slackbot.modal_app import rag_vol


class Scanner:

    def __init__(self, docs_dir: Path, get_indexed: Callable[[], dict[str, str]]):
        self._docs_dir = docs_dir
        self._get_indexed = get_indexed

    def scan(self) -> list[str]:
        """Compare disk fingerprints against ChromaDB, return new/changed files."""
        rag_vol.reload()
        if not self._docs_dir.exists():
            return []

        # {filename: fingerprint} for files already in ChromaDB
        indexed = self._get_indexed()

        # Compare each file's current fingerprint against what's indexed
        all_files = sorted(p for p in self._docs_dir.iterdir() if p.is_file())
        new_or_changed = [p for p in all_files if indexed.get(p.name) != self._fingerprint(p)]

        self._log(new_or_changed, len(all_files))
        return [str(p) for p in new_or_changed]

    def _fingerprint(self, path: Path) -> str:
        stat = path.stat()
        return f"{stat.st_mtime_ns}:{stat.st_size}"

    def _log(self, to_index: list[Path], total: int) -> None:
        print(f"[scan] {len(to_index)} to index, {total - len(to_index)} already indexed", flush=True)
        for p in to_index:
            print(f"[scan]   {p.name}", flush=True)
