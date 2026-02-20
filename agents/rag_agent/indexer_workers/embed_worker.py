"""GPU embedding worker — model loads once, concurrent inputs share it."""

import modal

from agents.slackbot.shared import app, rag_vol

from .config import EMBED_BATCH_SIZE, WORKERS_PER_GPU, hf_secret, index_image


@app.cls(
    image=index_image,
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
        import os
        import pickle

        all_records = []
        for file_path in file_paths:
            source = os.path.basename(file_path)
            print(f"  worker-{worker_id}: processing {source}...", flush=True)
            self._process_file(file_path, source, worker_id, all_records)
            print(f"  worker-{worker_id}: {source} done — {len(all_records):,} chunks total", flush=True)

        pkl_path = self._pending_dir / f"worker-{worker_id}.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(all_records, f)
        rag_vol.commit()

        print(f"  worker-{worker_id}: wrote {len(all_records):,} chunks to {pkl_path.name}", flush=True)
        return worker_id, len(all_records)

    def _process_file(self, file_path: str, source: str, worker_id: int, records: list):
        """Parse a single file (zip or regular) and append embedded records."""
        if file_path.endswith(".zip"):
            self._process_zip(file_path, source, worker_id, records)
        else:
            from llama_index.core import SimpleDirectoryReader

            docs = SimpleDirectoryReader(input_files=[file_path]).load_data()
            for doc in docs:
                doc.metadata["source"] = source
            self._embed_batch(docs, records)

    def _process_zip(self, file_path: str, source: str, worker_id: int, records: list):
        """Stream zip entries in batches, embedding each batch."""
        import zipfile

        from llama_index.core import Document

        with zipfile.ZipFile(file_path) as zf:
            names = [n for n in zf.namelist() if not n.endswith("/")]
            batch = []
            for i, name in enumerate(names):
                text = self._read_zip_entry(zf, name)
                if not text or not text.strip():
                    continue
                batch.append(Document(text=text, metadata={"source": source, "filename": name}))
                if len(batch) >= EMBED_BATCH_SIZE:
                    self._embed_batch(batch, records)
                    batch = []
                    if (i + 1) % (EMBED_BATCH_SIZE * 10) == 0:
                        print(f"  worker-{worker_id}: {source} {i + 1}/{len(names)} entries, {len(records):,} chunks", flush=True)
            if batch:
                self._embed_batch(batch, records)

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

    def _embed_batch(self, docs: list, records: list):
        """Chunk documents, embed on GPU (serialized via lock), append records."""
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
