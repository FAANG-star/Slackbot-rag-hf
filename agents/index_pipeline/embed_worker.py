"""GPU embedding worker — embeds documents and upserts to ChromaDB."""

import io
import json
import threading
import zipfile
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
    MODEL_BATCH_SIZE,
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
    env={
        "HF_HOME": "/data/hf-cache",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "WORKER_VERSION": "7",
    },
)
@modal.concurrent(max_inputs=WORKERS_PER_GPU)
class EmbedWorker:
    """GPU worker that embeds documents and upserts to ChromaDB.

    Accepts structured work items:
      - {"type": "files", "paths": [...]} — regular files, one per item
      - {"type": "zip_entries", "zip_path": "...", "entries": [...]} — a slice of
        zip entry names so large zips are split across N_WORKERS containers

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
    def embed(self, work: dict, worker_id: int = 0) -> tuple[int, int]:
        """Embed a work item and upsert to ChromaDB. Returns (worker_id, chunk_count).

        work is either:
          {"type": "files", "paths": [...]}
          {"type": "zip_entries", "zip_path": "...", "entries": [...]}
        """
        self._worker_manifest = {}
        self._worker_manifest_path = _CHROMA_DIR / f"manifest-worker-{worker_id}.json"

        work_type = work["type"]
        if work_type == "files":
            total = self._embed_files(work["paths"], worker_id)
        elif work_type == "zip_entries":
            total = self._embed_zip_entries(work["zip_path"], work["entries"], worker_id)
        else:
            raise ValueError(f"Unknown work type: {work_type}")

        return worker_id, total

    # ── File processing ─────────────────────────────────────────────────

    def _embed_files(self, file_paths: list[str], worker_id: int) -> int:
        total = 0
        for file_path in file_paths:
            source = Path(file_path).name
            print(f"  worker-{worker_id}: processing {source}...", flush=True)
            docs = self._load_file(file_path, source)
            count = self._embed_and_upsert(docs)
            total += count
            print(f"  worker-{worker_id}: {source} done — {total:,} chunks total", flush=True)
            self._update_manifest(Path(file_path))
        return total

    def _load_file(self, file_path: str, source: str) -> list:
        from llama_index.core import SimpleDirectoryReader
        docs = SimpleDirectoryReader(input_files=[file_path]).load_data()
        for doc in docs:
            doc.metadata["source"] = source
        return docs

    def _embed_zip_entries(self, zip_path: str, entries: list[str], worker_id: int) -> int:
        """Embed a slice of zip entries. Called once per worker — no concurrent zip access."""
        from llama_index.core import Document

        source = Path(zip_path).name
        print(f"  worker-{worker_id}: processing {len(entries):,} entries from {source}...", flush=True)

        total = 0
        with zipfile.ZipFile(zip_path) as zf:
            batch: list[Document] = []
            for i, name in enumerate(entries):
                text = self._read_zip_entry(zf, name)
                if not text or not text.strip():
                    continue
                batch.append(Document(text=text, metadata={"source": source, "filename": name}))
                if len(batch) >= EMBED_BATCH_SIZE:
                    total += self._embed_and_upsert(batch)
                    batch = []
                    if (i + 1) % (EMBED_BATCH_SIZE * 10) == 0:
                        print(f"    worker-{worker_id}: {i + 1}/{len(entries)} entries, {total:,} chunks", flush=True)
            if batch:
                total += self._embed_and_upsert(batch)

        print(f"  worker-{worker_id}: done — {total:,} chunks", flush=True)
        # Manifest keyed on zip filename — last worker to finish wins, that's fine
        # (finalize merges all worker manifests anyway)
        self._update_manifest(Path(zip_path))
        return total

    def _read_zip_entry(self, zf: zipfile.ZipFile, name: str) -> str | None:
        try:
            data = zf.read(name)
            if name.lower().endswith(".pdf"):
                from pypdf import PdfReader
                reader = PdfReader(io.BytesIO(data))
                return "\n".join(page.extract_text() or "" for page in reader.pages)
            return data.decode("utf-8", errors="replace")
        except Exception:
            return None

    # ── Embedding pipeline ──────────────────────────────────────────────

    def _embed_and_upsert(self, docs: list) -> int:
        """Chunk → embed → upsert. Returns number of chunks produced."""
        nodes = self._splitter.get_nodes_from_documents(docs)
        if not nodes:
            return 0
        texts = [n.get_content() for n in nodes]
        with self._gpu_lock:
            embeddings = self._embed_model.get_text_embedding_batch(texts, show_progress=False)
        for i in range(0, len(nodes), UPSERT_BATCH):
            batch = list(zip(nodes, embeddings, texts))[i : i + UPSERT_BATCH]
            with self._chroma_lock:
                self._collection.upsert(
                    ids=[n.node_id for n, _, _ in batch],
                    embeddings=[e for _, e, _ in batch],
                    documents=[t for _, _, t in batch],
                    metadatas=[n.metadata for n, _, _ in batch],
                )
        return len(nodes)

    # ── Setup ───────────────────────────────────────────────────────────

    def _init_models(self) -> None:
        from llama_index.core.node_parser import TokenTextSplitter
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding

        self._embed_model = HuggingFaceEmbedding(
            model_name=EMBEDDING_MODEL, device="cuda",
            normalize=True, embed_batch_size=MODEL_BATCH_SIZE,
            model_kwargs={"torch_dtype": "float16"},
        )
        self._splitter = TokenTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    def _init_chroma(self) -> None:
        import chromadb

        _CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        self._collection = client.get_or_create_collection(CHROMA_COLLECTION)

    # ── Manifest tracking ───────────────────────────────────────────────

    def _update_manifest(self, path: Path) -> None:
        """Record a successfully indexed file in this worker's manifest."""
        stat = path.stat()
        with self._manifest_lock:
            self._worker_manifest[path.name] = f"{stat.st_mtime_ns}:{stat.st_size}"
            self._worker_manifest_path.write_text(json.dumps(self._worker_manifest, indent=2))
            rag_vol.commit()
