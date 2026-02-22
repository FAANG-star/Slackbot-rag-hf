"""CPU upsert worker — receives chunks from GPU workers and writes to ChromaDB."""

import json
import queue
import threading
from pathlib import Path

import modal

from agents.slackbot.infra import app, rag_vol
from .. import config

Chunk = tuple[str, list[float], str, dict]  # (id, embedding, text, metadata)

_CHROMA_DIR = Path(config.CHROMA_DIR)


@app.cls(
    image=config.index_image,
    volumes={"/data": rag_vol},
    scaledown_window=30 * 60,
)
@modal.concurrent(max_inputs=64)
class UpsertWorker:
    """Receives embedded chunks from GPU workers and upserts them to ChromaDB serially."""

    @modal.enter()
    def _setup(self):
        import chromadb

        _CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=config.CHROMA_DIR)
        self._collection = client.get_or_create_collection(config.CHROMA_COLLECTION)
        self._queue: queue.Queue = queue.Queue()
        self._drain_thread = threading.Thread(target=self._drain, daemon=True)
        self._drain_thread.start()

    @modal.method()
    def receive(self, chunks: list, worker_id: int, work: dict) -> None:
        """Enqueue chunks for serial upsert. Returns immediately."""
        self._queue.put((chunks, worker_id, work))

    @modal.method()
    def flush(self) -> None:
        """Block until all enqueued batches have been upserted. Call after all workers finish."""
        done = threading.Event()
        self._queue.put(done)
        done.wait()

    @modal.method()
    def reset(self) -> None:
        """Drop and recreate the collection (for force reindex)."""
        import chromadb
        client = chromadb.PersistentClient(path=config.CHROMA_DIR)
        try:
            client.delete_collection(config.CHROMA_COLLECTION)
        except Exception:
            pass
        for f in _CHROMA_DIR.glob("manifest*.json"):
            f.unlink()
        self._collection = client.get_or_create_collection(config.CHROMA_COLLECTION)
        rag_vol.commit()

    def _drain(self) -> None:
        """Serial upsert loop — runs in background thread."""
        while True:
            item = self._queue.get()
            if isinstance(item, threading.Event):
                item.set()
                continue
            chunks, worker_id, work = item
            self._upsert(chunks, worker_id)
            self._write_manifest(worker_id, work)

    def _upsert(self, chunks: list[Chunk], worker_id: int) -> None:
        if not chunks:
            return
        print(f"  upsert-worker: upserting {len(chunks):,} chunks from worker-{worker_id}...", flush=True)
        for i in range(0, len(chunks), config.UPSERT_BATCH):
            batch = chunks[i : i + config.UPSERT_BATCH]
            ids, embeddings, documents, metadatas = zip(*batch)
            self._collection.upsert(
                ids=list(ids),
                embeddings=list(embeddings),
                documents=list(documents),
                metadatas=list(metadatas),
            )

    def _write_manifest(self, worker_id: int, work: dict) -> None:
        manifest = {}
        paths = work.get("paths", [work["zip_path"]] if "zip_path" in work else [])
        for p in paths:
            path = Path(p)
            stat = path.stat()
            manifest[path.name] = f"{stat.st_mtime_ns}:{stat.st_size}"
        manifest_path = _CHROMA_DIR / f"manifest-worker-{worker_id}.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))
        rag_vol.commit()
        print(f"  upsert-worker: committed worker-{worker_id}", flush=True)
