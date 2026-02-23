"""ChromaDB upsert + collection management."""

from pathlib import Path

UPSERT_BATCH = 5_000


class ChunkStore:

    def __init__(self, chroma_dir: str, collection_name: str):
        import chromadb

        Path(chroma_dir).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=chroma_dir)
        self._collection_name = collection_name
        self._collection = self._client.get_or_create_collection(collection_name)

    def upsert(self, chunks: list, worker_id: int) -> None:
        if not chunks:
            return
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

    def reset(self) -> None:
        try:
            self._client.delete_collection(self._collection_name)
        except Exception:
            pass
        self._collection = self._client.get_or_create_collection(self._collection_name)

    def count(self) -> int:
        return self._collection.count()
