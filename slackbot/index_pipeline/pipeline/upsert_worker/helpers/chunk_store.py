"""ChromaDB upsert + collection management."""

from pathlib import Path

UPSERT_BATCH = 5_000


class ChunkStore:

    def __init__(self, chroma_dir: str, collection_name: str):
        import chromadb

        Path(chroma_dir).mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=chroma_dir)
        self._collection = client.get_or_create_collection(collection_name)

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

    def get_indexed_files(self) -> dict[str, str]:
        """Return {source: fingerprint} for all indexed files."""
        result = self._collection.get(include=["metadatas"])
        indexed: dict[str, str] = {}
        for meta in result["metadatas"] or []:
            source = meta.get("source")
            fingerprint = meta.get("fingerprint")
            if source and fingerprint:
                indexed[source] = fingerprint
        return indexed
