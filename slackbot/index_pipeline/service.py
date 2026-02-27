"""Orchestrates scan → parallel embed → upsert → summary."""

from pathlib import Path

from .pipeline.embed_worker import EmbedWorker, WORKERS_PER_GPU
from .pipeline.upsert_worker import UpsertWorker
from .pipeline.preprocess.batch_builder import BatchBuilder
from .pipeline.preprocess.scanner import Scanner

# Number of GPU containers to fan out across
N_WORKERS = 8


class IndexService:
    """Scans for new docs, embeds in parallel on GPU, upserts to ChromaDB."""

    def __init__(self, docs_dir: Path = Path("/data/rag/docs")):
        self._embed_worker = EmbedWorker()
        self._upsert_worker = UpsertWorker()
        self._scanner = Scanner(docs_dir, self._upsert_worker.get_indexed_files.remote)
        self._batch_builder = BatchBuilder(N_WORKERS * WORKERS_PER_GPU)

    def index(self) -> str:
        """Run the full indexing pipeline. Blocks until complete."""

        # Find new/changed files by comparing disk fingerprints to ChromaDB
        files = self._scanner.scan()

        # Split files into per-worker batches (N_WORKERS × WORKERS_PER_GPU)
        batches = self._batch_builder.build(files)

        # Embed on GPU, upsert to ChromaDB as each embed finishes
        embeddings = self._embed_worker.embed.starmap(batches, order_outputs=False)
        
        chunks = sum(self._upsert_worker.upsert.remote(r) for r in embeddings)

        return f"Indexed {chunks:,} passages."
