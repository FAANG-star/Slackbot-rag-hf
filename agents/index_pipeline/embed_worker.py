"""GPU embedding worker — embeds documents and upserts to ChromaDB."""

import json
import threading
from pathlib import Path

import modal

from agents.slackbot.infra import app, rag_vol

from .config import (
    CHROMA_COLLECTION,
    CHROMA_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBED_BATCH_SIZE,
    EMBEDDING_MODEL,
    UPSERT_BATCH,
    WORKERS_PER_GPU,
    hf_secret,
    index_image,
)

_CHROMA_DIR = Path(CHROMA_DIR)


@app.cls(
    image=index_image,
    volumes={"/data": rag_vol},
    secrets=[hf_secret],
    gpu="T4",
    timeout=60 * 60,
    env={"HF_HOME": "/data/hf-cache"},
)
@modal.concurrent(max_inputs=WORKERS_PER_GPU)
class EmbedWorker:
    """GPU worker that embeds documents and upserts to ChromaDB.

    Pipeline per file:
      parse (file_parser) → chunk (SentenceSplitter) → embed (GPU) → upsert (ChromaDB)

    Each worker writes its own manifest file (manifest-worker-{id}.json) to avoid
    cross-container races. The finalize step merges these into manifest.json.
    """

    @modal.enter()
    def _setup(self):
        self._gpu_lock = threading.Lock()
        self._chroma_lock = threading.Lock()
        self._manifest_lock = threading.Lock()
        self._worker_manifest: dict = {}
        self._worker_manifest_path = _CHROMA_DIR / "manifest-worker-0.json"
        self._init_models()
        self._init_chroma()

    # ── Public API ──────────────────────────────────────────────────────

    @modal.method()
    def embed(self, file_paths: list[str], worker_id: int = 0) -> tuple[int, int]:
        """Embed files and upsert to ChromaDB. Returns (worker_id, chunk_count)."""
        from .file_parser import parse_file, parse_zip_batched

        self._worker_manifest = {}
        self._worker_manifest_path = _CHROMA_DIR / f"manifest-worker-{worker_id}.json"

        total = 0
        for file_path in file_paths:
            source = Path(file_path).name
            print(f"  worker-{worker_id}: processing {source}...", flush=True)
            if file_path.endswith(".zip"):
                for batch in parse_zip_batched(file_path, source, EMBED_BATCH_SIZE):
                    total += self._chunk_embed_upsert(batch)
            else:
                total += self._chunk_embed_upsert(parse_file(file_path, source))
            print(f"  worker-{worker_id}: {source} done — {total:,} chunks total", flush=True)
            self._update_manifest(Path(file_path))

        return worker_id, total

    # ── Setup ───────────────────────────────────────────────────────────

    def _init_models(self) -> None:
        from llama_index.core.node_parser import SentenceSplitter
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding

        self._embed_model = HuggingFaceEmbedding(
            model_name=EMBEDDING_MODEL, device="cuda",
            normalize=True, embed_batch_size=EMBED_BATCH_SIZE,
        )
        self._splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    def _init_chroma(self) -> None:
        import chromadb

        _CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        self._collection = client.get_or_create_collection(CHROMA_COLLECTION)

    # ── Embedding pipeline ──────────────────────────────────────────────

    def _chunk_embed_upsert(self, docs: list) -> int:
        """Chunk → embed → upsert. Returns number of chunks produced."""
        nodes = self._splitter.get_nodes_from_documents(docs)
        if not nodes:
            return 0
        texts = [n.get_content() for n in nodes]
        embeddings = self._embed_texts(texts)
        self._upsert(nodes, embeddings, texts)
        rag_vol.commit()
        return len(nodes)

    def _embed_texts(self, texts: list[str]) -> list:
        with self._gpu_lock:
            return self._embed_model.get_text_embedding_batch(texts, show_progress=False)

    def _upsert(self, nodes, embeddings, texts) -> None:
        for i in range(0, len(nodes), UPSERT_BATCH):
            batch = list(zip(nodes, embeddings, texts))[i : i + UPSERT_BATCH]
            with self._chroma_lock:
                self._collection.upsert(
                    ids=[n.node_id for n, _, _ in batch],
                    embeddings=[e for _, e, _ in batch],
                    documents=[t for _, _, t in batch],
                    metadatas=[n.metadata for n, _, _ in batch],
                )

    # ── Manifest tracking ───────────────────────────────────────────────

    def _update_manifest(self, path: Path) -> None:
        """Record a successfully indexed file in this worker's manifest."""
        stat = path.stat()
        with self._manifest_lock:
            self._worker_manifest[path.name] = f"{stat.st_mtime_ns}:{stat.st_size}"
            self._worker_manifest_path.write_text(json.dumps(self._worker_manifest, indent=2))
        rag_vol.commit()
