"""CPU upsert worker — writes chunks to ChromaDB."""

import modal

from slackbot.app import app, rag_vol
from slackbot.index_pipeline import config

from .helpers.chunk_store import ChunkStore


@app.cls(
    image=config.upsert_image,
    volumes={"/data": rag_vol},
    scaledown_window=30 * 60,
    timeout=60 * 60,
)
@modal.concurrent(max_inputs=64)
class UpsertWorker:

    @modal.enter()
    def _setup(self):
        self._store = ChunkStore(config.CHROMA_DIR, config.CHROMA_COLLECTION)

    @modal.method()
    def upsert(self, chunks: list, worker_id: int) -> int:
        """Write chunks to ChromaDB. Returns number of chunks written."""
        self._store.upsert(chunks, worker_id)
        return len(chunks)

    @modal.method()
    def get_indexed_files(self) -> dict[str, str]:
        """Return {source: fingerprint} for all indexed files."""
        return self._store.get_indexed_files()
