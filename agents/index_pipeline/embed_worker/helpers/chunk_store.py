"""ChromaDB bulk upsert and manifest tracking."""

import json
import threading
from pathlib import Path

import modal

from ...config import UPSERT_BATCH

Chunk = tuple[str, list[float], str, dict]  # (id, embedding, text, metadata)


class ChunkStore:

    def __init__(self, chroma_dir: str, collection_name: str, volume: modal.Volume):
        import chromadb

        self._chroma_dir = Path(chroma_dir)
        self._volume = volume
        self._lock = threading.Lock()

        self._chroma_dir.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=chroma_dir)
        self._collection = client.get_or_create_collection(collection_name)

    def flush(self, pending: list[Chunk], worker_id: int) -> None:
        if not pending:
            return
        print(f"  worker-{worker_id}: upserting {len(pending):,} chunks...", flush=True)
        with self._lock:
            for i in range(0, len(pending), UPSERT_BATCH):
                batch = pending[i : i + UPSERT_BATCH]
                ids, embeddings, documents, metadatas = zip(*batch)
                self._collection.upsert(
                    ids=list(ids),
                    embeddings=list(embeddings),
                    documents=list(documents),
                    metadatas=list(metadatas),
                )

    def write_manifest(self, worker_id: int, work: dict) -> None:
        manifest = {}
        paths = work.get("paths", [work["zip_path"]] if "zip_path" in work else [])
        for p in paths:
            path = Path(p)
            stat = path.stat()
            manifest[path.name] = f"{stat.st_mtime_ns}:{stat.st_size}"

        manifest_path = self._chroma_dir / f"manifest-worker-{worker_id}.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))
        self._volume.commit()
        print(f"  worker-{worker_id}: committed", flush=True)
