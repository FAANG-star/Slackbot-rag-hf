"""Splits files and zip entries into per-worker batches."""

import math
import zipfile


class BatchBuilder:

    def __init__(self, n_batches: int):
        self._n = n_batches

    def build(self, files: list[str]) -> list[tuple[dict, int]]:
        """Return (work_dict, worker_id) tuples ready for embed.starmap()."""
        batches: list[dict] = []
        batches.extend(self._split_files(files))
        batches.extend(self._split_zips(files))
        return [(batch, i) for i, batch in enumerate(batches)]

    def _split_files(self, files: list[str]) -> list[dict]:
        """Distribute regular files evenly across workers."""
        regular = [f for f in files if not f.endswith(".zip")]
        return [{"type": "files", "paths": chunk} for chunk in self._chunk(regular)]

    def _split_zips(self, files: list[str]) -> list[dict]:
        """Expand each zip and distribute its entries across workers."""
        batches = []
        for path in files:
            if not path.endswith(".zip"):
                continue
            with zipfile.ZipFile(path) as zf:
                entries = [n for n in zf.namelist() if not n.endswith("/")]
            for chunk in self._chunk(entries):
                batches.append({"type": "zip_entries", "zip_path": path, "entries": chunk})
        return batches

    def _chunk(self, items: list) -> list[list]:
        """Split items into roughly equal groups (up to self._n).

        Each chunk is ceil(len(items) / n_batches) items. With 8 GPUs × 4
        workers per GPU = 32 batches, 100 files produces 32 chunks of ~4 each.
        """
        if not items:
            return []
        size = math.ceil(len(items) / self._n)
        return [items[i : i + size] for i in range(0, len(items), size)]
