"""Wraps a ChromaDB collection for chunk-level operations."""

import chromadb

from ..config import CHROMA_COLLECTION


class ChunkStore:
    def __init__(self, client: chromadb.ClientAPI):
        self._client = client
        self.collection = client.get_or_create_collection(CHROMA_COLLECTION)

    def delete_by_source(self, filenames: set[str]) -> int:
        """Delete all chunks matching the given source filenames."""
        deleted = 0
        for fname in filenames:
            results = self.collection.get(where={"source": fname})
            if results["ids"]:
                self.collection.delete(ids=results["ids"])
                deleted += len(results["ids"])
        return deleted

    def count(self) -> int:
        return self.collection.count()

    def drop(self):
        """Drop and recreate the collection."""
        try:
            self._client.delete_collection(CHROMA_COLLECTION)
        except Exception:
            pass
        self.collection = self._client.get_or_create_collection(CHROMA_COLLECTION)
