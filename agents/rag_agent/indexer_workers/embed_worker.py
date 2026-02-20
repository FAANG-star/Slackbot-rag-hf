"""GPU embedding worker — embeds and writes directly to ChromaDB per batch."""

import modal

from agents.slackbot.shared import app, rag_vol

from .config import EMBED_BATCH_SIZE, WORKERS_PER_GPU, hf_secret, index_image

CHROMA_DIR = "/data/rag/chroma"
CHROMA_COLLECTION = "rag_documents"
UPSERT_BATCH = 5_000


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
    """GPU worker — embeds documents and upserts directly to ChromaDB in batches."""

    @modal.enter()
    def setup(self):
        import json
        import threading
        from pathlib import Path

        import chromadb
        from llama_index.core.node_parser import SentenceSplitter
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding

        self._embed_model = HuggingFaceEmbedding(
            model_name="BAAI/bge-large-en-v1.5",
            device="cuda",
            normalize=True,
            embed_batch_size=EMBED_BATCH_SIZE,
        )
        self._splitter = SentenceSplitter(chunk_size=512, chunk_overlap=64)
        self._gpu_lock = threading.Lock()
        self._chroma_lock = threading.Lock()

        self._chroma_dir = Path(CHROMA_DIR)
        self._chroma_dir.mkdir(parents=True, exist_ok=True)
        self._manifest_lock = threading.Lock()
        # worker_manifest_path is set per-invocation in embed() — each worker_id
        # writes its own file to avoid cross-container overwrite races.
        self._worker_manifest: dict = {}
        self._worker_manifest_path: Path | None = None
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        self._collection = client.get_or_create_collection(CHROMA_COLLECTION)

    @modal.method()
    def embed(self, file_paths: list[str], worker_id: int = 0) -> tuple[int, int]:
        """Embed files and write directly to ChromaDB. Returns (worker_id, chunk_count)."""
        import os
        from pathlib import Path

        # Each worker writes its own manifest file — no cross-container overwrite race.
        self._worker_manifest = {}
        self._worker_manifest_path = self._chroma_dir / f"manifest-worker-{worker_id}.json"

        total = 0
        for file_path in file_paths:
            source = os.path.basename(file_path)
            print(f"  worker-{worker_id}: processing {source}...", flush=True)
            count = self._process_file(file_path, source, worker_id)
            total += count
            print(f"  worker-{worker_id}: {source} done — {total:,} chunks total", flush=True)
            self._record_manifest(Path(file_path))

        return worker_id, total

    def _record_manifest(self, path):
        """Append a successfully indexed file to this worker's manifest file."""
        import json

        stat = path.stat()
        with self._manifest_lock:
            self._worker_manifest[path.name] = f"{stat.st_mtime_ns}:{stat.st_size}"
            self._worker_manifest_path.write_text(json.dumps(self._worker_manifest, indent=2))
        rag_vol.commit()

    def _process_file(self, file_path: str, source: str, worker_id: int) -> int:
        """Parse a single file (zip or regular), embed and upsert. Returns chunk count."""
        if file_path.endswith(".zip"):
            return self._process_zip(file_path, source, worker_id)
        from llama_index.core import SimpleDirectoryReader

        docs = SimpleDirectoryReader(input_files=[file_path]).load_data()
        for doc in docs:
            doc.metadata["source"] = source
        return self._embed_and_upsert(docs, worker_id)

    def _process_zip(self, file_path: str, source: str, worker_id: int) -> int:
        """Stream zip entries in batches, embedding and upserting each batch."""
        import zipfile

        from llama_index.core import Document

        total = 0
        with zipfile.ZipFile(file_path) as zf:
            names = [n for n in zf.namelist() if not n.endswith("/")]
            batch = []
            for i, name in enumerate(names):
                text = self._read_zip_entry(zf, name)
                if not text or not text.strip():
                    continue
                batch.append(Document(text=text, metadata={"source": source, "filename": name}))
                if len(batch) >= EMBED_BATCH_SIZE:
                    total += self._embed_and_upsert(batch, worker_id)
                    batch = []
                    if (i + 1) % (EMBED_BATCH_SIZE * 10) == 0:
                        print(f"  worker-{worker_id}: {source} {i + 1}/{len(names)} entries, {total:,} chunks", flush=True)
            if batch:
                total += self._embed_and_upsert(batch, worker_id)
        return total

    def _read_zip_entry(self, zf, name: str) -> str | None:
        """Read a single zip entry as text. Returns None on failure."""
        import io

        from pypdf import PdfReader

        try:
            data = zf.read(name)
            if name.lower().endswith(".pdf"):
                reader = PdfReader(io.BytesIO(data))
                return "\n".join(page.extract_text() or "" for page in reader.pages)
            return data.decode("utf-8", errors="replace")
        except Exception:
            return None

    def _embed_and_upsert(self, docs: list, worker_id: int) -> int:
        """Chunk, embed, and upsert a batch of docs. Returns chunk count."""
        nodes = self._splitter.get_nodes_from_documents(docs)
        if not nodes:
            return 0
        texts = [n.get_content() for n in nodes]
        with self._gpu_lock:
            embeddings = self._embed_model.get_text_embedding_batch(texts, show_progress=False)
        records = [
            {"id": n.node_id, "embedding": emb, "document": txt, "metadata": n.metadata}
            for n, emb, txt in zip(nodes, embeddings, texts)
        ]
        for i in range(0, len(records), UPSERT_BATCH):
            chunk = records[i : i + UPSERT_BATCH]
            with self._chroma_lock:
                self._collection.upsert(
                    ids=[r["id"] for r in chunk],
                    embeddings=[r["embedding"] for r in chunk],
                    documents=[r["document"] for r in chunk],
                    metadatas=[r["metadata"] for r in chunk],
                )
        rag_vol.commit()
        return len(records)
