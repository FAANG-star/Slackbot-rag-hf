"""Read-only vector index for query-time search."""

import chromadb
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from ..config import CHROMA_COLLECTION, CHROMA_DIR, EMBEDDING_MODEL


class SearchIndex:
    """Holds the embedding model and VectorStoreIndex for query-time search.

    The index_pipeline handles writing to ChromaDB — this class only reads.
    """

    def __init__(self):
        self.embed_model = HuggingFaceEmbedding(
            model_name=EMBEDDING_MODEL,
            device="cuda",
            normalize=True,
            embed_batch_size=256,
        )
        self._load_collection()

    def _load_collection(self):
        """Open ChromaDB and build the LlamaIndex vector store."""
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self._collection = client.get_or_create_collection(CHROMA_COLLECTION)
        vector_store = ChromaVectorStore(chroma_collection=self._collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        self.index = VectorStoreIndex.from_vector_store(
            vector_store,
            embed_model=self.embed_model,
            storage_context=storage_context,
        )

    def reload(self):
        """Reconnect to ChromaDB to pick up changes from the index pipeline."""
        self._load_collection()

    def has_index(self) -> bool:
        """Check if any documents have been indexed."""
        return self._collection.count() > 0
