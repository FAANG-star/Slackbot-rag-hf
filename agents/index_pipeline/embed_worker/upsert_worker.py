"""CPU upsert worker — receives chunks from GPU workers and writes to ChromaDB.

Flow: begin(n) → n × receive() via .spawn() → drain thread auto-finalizes after nth batch.
Poll progress() for (drained, expected, result_or_none).
"""

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
    timeout=60 * 60,
)
@modal.concurrent(max_inputs=64)
class UpsertWorker:

    @modal.enter()
    def _setup(self):
        import chromadb

        _CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=config.CHROMA_DIR)
        self._collection = client.get_or_create_collection(config.CHROMA_COLLECTION)
        self._queue: queue.Queue = queue.Queue()
        self._expected = 0
        self._drained = 0
        self._result: str | None = None
        self._drain_thread = threading.Thread(target=self._drain, daemon=True)
        self._drain_thread.start()

    @modal.method()
    def begin(self, n_batches: int) -> None:
        """Tell the worker how many receive() calls to expect."""
        self._expected = n_batches
        self._drained = 0
        self._result = None

    @modal.method()
    def receive(self, chunks: list, worker_id: int, work: dict) -> None:
        """Enqueue chunks for serial upsert and return immediately."""
        self._queue.put((chunks, worker_id, work))

    @modal.method()
    def progress(self) -> tuple[int, int, str | None]:
        """Return (drained, expected, result_or_none). Instant call."""
        return self._drained, self._expected, self._result

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

    # ── Internal ────────────────────────────────────────────────────────

    def _drain(self) -> None:
        """Serial upsert loop — auto-finalizes after expected batch count."""
        while True:
            chunks, worker_id, work = self._queue.get()
            self._upsert(chunks, worker_id)
            self._write_manifest(worker_id, work)
            self._drained += 1
            if self._expected and self._drained >= self._expected:
                self._finalize()

    def _finalize(self) -> None:
        import chromadb

        print("upsert-worker: finalizing index...", flush=True)
        manifest = self._merge_manifests()
        count = chromadb.PersistentClient(path=config.CHROMA_DIR).get_or_create_collection(
            config.CHROMA_COLLECTION
        ).count()
        files = f"{len(manifest)} file{'s' if len(manifest) != 1 else ''}"
        self._result = f"Done! {count:,} searchable passages from {files}."
        print(f"upsert-worker: {self._result}", flush=True)

    def _merge_manifests(self) -> dict:
        try:
            manifest = json.loads((_CHROMA_DIR / "manifest.json").read_text())
        except FileNotFoundError:
            manifest = {}
        for wf in sorted(_CHROMA_DIR.glob("manifest-worker-*.json")):
            manifest.update(json.loads(wf.read_text()))
            wf.unlink()
        (_CHROMA_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
        rag_vol.commit()
        return manifest

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
