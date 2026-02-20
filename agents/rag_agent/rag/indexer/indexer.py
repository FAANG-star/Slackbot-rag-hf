"""Composes Manifest, ChunkStore, and DocumentParser to manage the RAG index."""

import chromadb
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from ..config import CHROMA_DIR, CHUNK_SIZE, EMBED_BATCH_SIZE, EMBEDDING_MODEL
from .chunk_store import ChunkStore
from .document_parser import DocumentParser
from .manifest import Manifest


class Indexer:
    """Create once at startup — holds the embedding model, ChromaDB client,
    and VectorStoreIndex for the process lifetime.
    """

    def __init__(self):
        self.embed_model = HuggingFaceEmbedding(
            model_name=EMBEDDING_MODEL,
            device="cuda",
            normalize=True,
            embed_batch_size=EMBED_BATCH_SIZE,
        )
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.manifest = Manifest()
        self.store = ChunkStore(self._client)
        self.parser = DocumentParser()
        self._rebuild_index()

    # ── Public API ───────────────────────────────────────────────────────────

    def reload(self):
        """Reload ChromaDB from volume without re-loading the embedding model."""
        self._client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.store = ChunkStore(self._client)
        self.manifest = Manifest()
        self._rebuild_index()

    def has_index(self) -> bool:
        return bool(self.manifest.load())

    def reset(self):
        """Drop the ChromaDB collection and rebuild an empty index."""
        self.store.drop()
        self._rebuild_index()

    def build(self, force: bool = False) -> tuple[int, int, int]:
        """Parse docs, chunk, embed, and store in ChromaDB.

        Returns:
            (total_chunks, newly_added, deleted_count)
        """
        current_files = self.manifest.scan()
        if not current_files:
            return (0, 0, 0)

        old = {} if force else self.manifest.load()
        if force:
            self.reset()

        to_add, to_delete = self.manifest.diff(old, current_files)
        deleted_count = self.store.delete_by_source(to_delete)
        added_count = self._insert_nodes(to_add)
        self.manifest.save(current_files)
        return (self.store.count(), added_count, deleted_count)

    def stats(self) -> str:
        """Return a summary of the current index."""
        files = self.manifest.load()
        if not files:
            return "No documents indexed yet."
        lines = [f"Index: {len(files)} source(s), {self.store.count()} chunks (ChromaDB)"]
        lines += [f"  - {fname}" for fname in sorted(files)]
        return "\n".join(lines)

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _rebuild_index(self):
        """(Re)create the VectorStoreIndex from the current collection."""
        vector_store = ChromaVectorStore(chroma_collection=self.store.collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        self.index = VectorStoreIndex.from_vector_store(
            vector_store,
            embed_model=self.embed_model,
            storage_context=storage_context,
        )

    def _insert_nodes(self, filenames: list[str]) -> int:
        """Parse and insert nodes for the given filenames. Returns count of inserted nodes."""
        added = 0
        pending = []
        for _, nodes in self.parser.parse(filenames):
            pending.extend(nodes)
            if len(pending) >= CHUNK_SIZE:
                self.index.insert_nodes(pending)
                added += len(pending)
                pending = []
        if pending:
            self.index.insert_nodes(pending)
            added += len(pending)
        return added
