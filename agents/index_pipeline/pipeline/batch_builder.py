"""Splits files and zip entries into per-worker batches."""

import math
import zipfile


class BatchBuilder:

    def __init__(self, n_batches: int):
        self._n = n_batches

    def build(self, files: list[str]) -> tuple[list[dict], int]:
        batches: list[dict] = []
        doc_count = 0

        # split regular files evenly across workers
        regular = [f for f in files if not f.endswith(".zip")]
        doc_count += len(regular)
        for chunk in self._chunk(regular):
            batches.append({"type": "files", "paths": chunk})

        # expand each zip's entries and distribute across workers
        for path in files:
            if not path.endswith(".zip"):
                continue
            with zipfile.ZipFile(path) as zf:
                entries = [n for n in zf.namelist() if not n.endswith("/")]
            doc_count += len(entries)
            for chunk in self._chunk(entries):
                batches.append({"type": "zip_entries", "zip_path": path, "entries": chunk})

        return batches, doc_count

    def _chunk(self, items: list) -> list[list]:
        if not items:
            return []
        size = math.ceil(len(items) / self._n)
        return [items[i : i + size] for i in range(0, len(items), size)]
