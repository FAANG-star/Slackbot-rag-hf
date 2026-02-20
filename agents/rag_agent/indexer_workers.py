"""Parallel indexing Modal functions — GPU workers for embedding, CPU for finalize."""

import modal

from agents.infra.shared import app, rag_vol

hf_secret = modal.Secret.from_name("hf-secret")

_index_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install(
        "llama-index-core",
        "llama-index-embeddings-huggingface",
        "llama-index-readers-file",
        "sentence-transformers>=3.0",
        "transformers>=4.51",
        "chromadb",
        "pypdf",
        "python-docx",
    )
)

EMBED_BATCH_SIZE = 1024
UPSERT_BATCH = 5_000
N_WORKERS = 10
WORKERS_PER_GPU = 4


@app.cls(
    image=_index_image,
    volumes={"/data": rag_vol},
    secrets=[hf_secret],
    gpu="A10G",
    timeout=60 * 60,
    env={"HF_HOME": "/data/hf-cache"},
)
@modal.concurrent(max_inputs=WORKERS_PER_GPU)
class EmbedWorker:
    """GPU worker — model loads once, multiple concurrent inputs share it."""

    @modal.enter()
    def setup(self):
        import threading
        from pathlib import Path

        from llama_index.core.node_parser import SentenceSplitter
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding

        self._embed_model = HuggingFaceEmbedding(
            model_name="BAAI/bge-large-en-v1.5",
            device="cuda",
            normalize=True,
            embed_batch_size=EMBED_BATCH_SIZE,
        )
        self._splitter = SentenceSplitter(chunk_size=512, chunk_overlap=64)
        self._pending_dir = Path("/data/rag/pending")
        self._pending_dir.mkdir(parents=True, exist_ok=True)
        self._gpu_lock = threading.Lock()

    @modal.method()
    def embed(self, file_paths: list[str], worker_id: int = 0) -> tuple[int, int]:
        """Embed files, write one pkl per worker. Returns (worker_id, chunk_count)."""
        import io
        import os
        import pickle
        import zipfile

        from llama_index.core import Document

        def _read_zip_text(zf, name):
            from pypdf import PdfReader

            try:
                data = zf.read(name)
                if name.lower().endswith(".pdf"):
                    reader = PdfReader(io.BytesIO(data))
                    return "\n".join(page.extract_text() or "" for page in reader.pages)
                return data.decode("utf-8", errors="replace")
            except Exception:
                return None

        def _embed_batch(docs, records):
            nodes = self._splitter.get_nodes_from_documents(docs)
            if not nodes:
                return
            texts = [n.get_content() for n in nodes]
            with self._gpu_lock:
                embeddings = self._embed_model.get_text_embedding_batch(texts, show_progress=False)
            records.extend(
                {"id": n.node_id, "embedding": emb, "document": txt, "metadata": n.metadata}
                for n, emb, txt in zip(nodes, embeddings, texts)
            )

        all_records = []
        for file_path in file_paths:
            source = os.path.basename(file_path)
            print(f"  worker-{worker_id}: processing {source}...", flush=True)

            if file_path.endswith(".zip"):
                with zipfile.ZipFile(file_path) as zf:
                    names = [n for n in zf.namelist() if not n.endswith("/")]
                    batch = []
                    for i, name in enumerate(names):
                        text = _read_zip_text(zf, name)
                        if not text or not text.strip():
                            continue
                        batch.append(Document(text=text, metadata={"source": source, "filename": name}))
                        if len(batch) >= EMBED_BATCH_SIZE:
                            _embed_batch(batch, all_records)
                            batch = []
                            if (i + 1) % (EMBED_BATCH_SIZE * 10) == 0:
                                print(f"  worker-{worker_id}: {source} {i + 1}/{len(names)} entries, {len(all_records):,} chunks", flush=True)
                    if batch:
                        _embed_batch(batch, all_records)
            else:
                from llama_index.core import SimpleDirectoryReader

                docs = SimpleDirectoryReader(input_files=[file_path]).load_data()
                for doc in docs:
                    doc.metadata["source"] = source
                _embed_batch(docs, all_records)

            print(f"  worker-{worker_id}: {source} done — {len(all_records):,} chunks total", flush=True)

        pkl_path = self._pending_dir / f"worker-{worker_id}.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(all_records, f)
        rag_vol.commit()

        print(f"  worker-{worker_id}: wrote {len(all_records):,} chunks to {pkl_path.name}", flush=True)
        return worker_id, len(all_records)


@app.function(image=_index_image, volumes={"/data": rag_vol}, timeout=60 * 60)
def finalize_index(force: bool = False) -> str:
    """Merge worker pkl files into ChromaDB and write manifest."""
    import json
    import pickle
    import shutil
    from pathlib import Path

    import chromadb

    CHROMA_DIR = Path("/data/rag/chroma")
    DOCS_DIR = Path("/data/rag/docs")
    PENDING_DIR = Path("/data/rag/pending")
    MANIFEST_PATH = CHROMA_DIR / "manifest.json"
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    if force:
        try:
            client.delete_collection("rag_documents")
        except Exception:
            pass
    collection = client.get_or_create_collection("rag_documents")

    pkl_files = sorted(PENDING_DIR.glob("*.pkl")) if PENDING_DIR.exists() else []
    print(f"Merging {len(pkl_files)} worker pkl(s) into ChromaDB...", flush=True)

    total = 0
    for pkl_path in pkl_files:
        with open(pkl_path, "rb") as f:
            records = pickle.load(f)
        for i in range(0, len(records), UPSERT_BATCH):
            chunk = records[i : i + UPSERT_BATCH]
            collection.upsert(
                ids=[r["id"] for r in chunk],
                embeddings=[r["embedding"] for r in chunk],
                documents=[r["document"] for r in chunk],
                metadatas=[r["metadata"] for r in chunk],
            )
        total += len(records)
        print(f"  {pkl_path.stem}: {len(records):,} chunks (total: {total:,})", flush=True)

    manifest = {}
    if DOCS_DIR.exists():
        for path in sorted(DOCS_DIR.iterdir()):
            if path.is_file():
                stat = path.stat()
                manifest[path.name] = f"{stat.st_mtime_ns}:{stat.st_size}"
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))

    if PENDING_DIR.exists():
        shutil.rmtree(PENDING_DIR)
    rag_vol.commit()

    summary = f"Indexed {total:,} chunks from {len(manifest)} file(s)."
    print(summary, flush=True)
    return summary
