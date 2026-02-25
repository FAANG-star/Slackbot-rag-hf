"""CPU upsert worker — writes embedded chunks to ChromaDB."""

from pathlib import Path
import modal
from slackbot.modal_app import app, rag_vol

CHROMA_DIR = "/data/rag/chroma"
CHROMA_COLLECTION = "rag_documents"
UPSERT_BATCH = 5_000

upsert_image = modal.Image.debian_slim(python_version="3.12").pip_install("chromadb")

@app.cls(
    image=upsert_image,
    volumes={"/data": rag_vol},
    timeout=60 * 60,
)
@modal.concurrent(max_inputs=1)
class UpsertWorker:

    @modal.enter()
    def _setup(self):
        import chromadb

        # SQLite-backed persistent store on the rag volume
        Path(CHROMA_DIR).mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        # Opens existing collection or creates a new empty one
        self._collection = client.get_or_create_collection(CHROMA_COLLECTION)

    @modal.method()
    def upsert(self, chunks: list, worker_id: int) -> int:
        """Write chunks to ChromaDB in batches of UPSERT_BATCH.

        Each chunk is (id, embedding, text, metadata) from EmbedWorker.
        """
        print(f"  upsert-worker: upserting {len(chunks):,} chunks from worker-{worker_id}...", flush=True)
        for i in range(0, len(chunks), UPSERT_BATCH):
            batch = chunks[i : i + UPSERT_BATCH]
            ids, embeddings, documents, metadatas = zip(*batch)
            self._collection.upsert(
                ids=list(ids),
                embeddings=list(embeddings),
                documents=list(documents),
                metadatas=list(metadatas),
            )
        return len(chunks)

    @modal.method()
    def get_indexed_files(self) -> dict[str, str]:
        """Return {source: fingerprint} for all indexed files.

        Used by Scanner to compare disk fingerprints against what's
        already in ChromaDB, skipping files that haven't changed.
        """
        indexed: dict[str, str] = {}
        total = self._collection.count()
        page_size = 5_000
        for offset in range(0, total, page_size):
            result = self._collection.get(include=["metadatas"], limit=page_size, offset=offset)
            for meta in result["metadatas"] or []:
                source = meta.get("source")
                fingerprint = meta.get("fingerprint")
                if source and fingerprint:
                    indexed[source] = fingerprint
        return indexed
