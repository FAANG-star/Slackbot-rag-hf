"""GPU embedding worker — embeds documents via TEI and upserts to ChromaDB."""

import io
import json
import socket
import subprocess
import threading
import time
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
    TEI_BATCH_SIZE,
    TEI_MAX_BATCH,
    TEI_PORT,
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
    gpu="A10G",
    timeout=60 * 60,
    env={
        "HF_HOME": "/data/hf-cache",
        "HUGGINGFACE_HUB_CACHE": "/data/hf-cache",
        "WORKER_VERSION": "15",
    },
)
@modal.concurrent(max_inputs=WORKERS_PER_GPU)
class EmbedWorker:
    """GPU worker that embeds documents via TEI and upserts to ChromaDB.

    Accepts structured work items:
      - {"type": "files", "paths": [...]} — regular files, one per item
      - {"type": "zip_entries", "zip_path": "...", "entries": [...]} — a slice of
        zip entry names so large zips are split across N_WORKERS containers

    Each worker writes its own manifest file (manifest-worker-{id}.json) to avoid
    cross-container races. The finalize step merges these into manifest.json.
    """

    @modal.enter()
    def _setup(self):
        self._chroma_lock = threading.Lock()
        self._start_tei()
        self._init_splitter()
        self._init_chroma()
        import httpx
        self._http = httpx.Client(timeout=120.0)

    # ── Public API ──────────────────────────────────────────────────────

    @modal.method()
    def embed(self, work: dict, worker_id: int = 0) -> tuple[int, int]:
        """Embed a work item and upsert to ChromaDB. Returns (worker_id, chunk_count).

        Accumulates all chunks in memory during embedding (GPU-bound phase),
        then does one bulk upsert + volume commit at the end to minimize GPU idle time.
        """
        self._pending: list[tuple[str, list[float], str, dict]] = []

        work_type = work["type"]
        if work_type == "files":
            total = self._embed_files(work["paths"], worker_id)
        elif work_type == "zip_entries":
            total = self._embed_zip_entries(work["zip_path"], work["entries"], worker_id)
        else:
            raise ValueError(f"Unknown work type: {work_type}")

        self._flush_to_chroma(worker_id)
        self._write_manifest(worker_id, work)
        return worker_id, total

    # ── File processing ─────────────────────────────────────────────────

    def _embed_files(self, file_paths: list[str], worker_id: int) -> int:
        total = 0
        for file_path in file_paths:
            source = Path(file_path).name
            print(f"  worker-{worker_id}: processing {source}...", flush=True)
            docs = self._load_file(file_path, source)
            count = self._chunk_and_embed(docs)
            total += count
            print(f"  worker-{worker_id}: {source} done — {total:,} chunks total", flush=True)
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
                    total += self._chunk_and_embed(batch)
                    batch = []
                    if (i + 1) % (EMBED_BATCH_SIZE * 10) == 0:
                        print(f"    worker-{worker_id}: {i + 1}/{len(entries)} entries, {total:,} chunks", flush=True)
            if batch:
                total += self._chunk_and_embed(batch)

        print(f"  worker-{worker_id}: done embedding {total:,} chunks", flush=True)
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

    def _chunk_and_embed(self, docs: list) -> int:
        """Chunk → embed via TEI → accumulate in memory. Returns chunk count."""
        nodes = self._splitter.get_nodes_from_documents(docs)
        if not nodes:
            return 0
        texts = [n.get_content() for n in nodes]
        embeddings = self._embed_texts(texts)
        for node, emb, text in zip(nodes, embeddings, texts):
            self._pending.append((node.node_id, emb, text, node.metadata))
        return len(nodes)

    def _flush_to_chroma(self, worker_id: int) -> None:
        """Bulk upsert all accumulated chunks to ChromaDB."""
        if not self._pending:
            return
        print(f"  worker-{worker_id}: upserting {len(self._pending):,} chunks to ChromaDB...", flush=True)
        with self._chroma_lock:
            for i in range(0, len(self._pending), UPSERT_BATCH):
                batch = self._pending[i : i + UPSERT_BATCH]
                self._collection.upsert(
                    ids=[id for id, _, _, _ in batch],
                    embeddings=[emb for _, emb, _, _ in batch],
                    documents=[doc for _, _, doc, _ in batch],
                    metadatas=[meta for _, _, _, meta in batch],
                )
        self._pending = []

    def _write_manifest(self, worker_id: int, work: dict) -> None:
        """Write manifest and commit volume once."""
        manifest = {}
        paths = work.get("paths", [work["zip_path"]] if "zip_path" in work else [])
        for p in paths:
            path = Path(p)
            stat = path.stat()
            manifest[path.name] = f"{stat.st_mtime_ns}:{stat.st_size}"
        manifest_path = _CHROMA_DIR / f"manifest-worker-{worker_id}.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))
        rag_vol.commit()
        print(f"  worker-{worker_id}: committed to volume", flush=True)

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed texts via TEI HTTP API, batched to avoid huge payloads."""
        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), TEI_BATCH_SIZE):
            batch = texts[i : i + TEI_BATCH_SIZE]
            resp = self._http.post(
                f"http://127.0.0.1:{TEI_PORT}/embed",
                json={"inputs": batch},
            )
            resp.raise_for_status()
            all_embeddings.extend(resp.json())
        return all_embeddings

    # ── TEI server ─────────────────────────────────────────────────────

    def _start_tei(self) -> None:
        """Launch TEI as a subprocess and wait until it's ready."""
        self._tei_proc = subprocess.Popen(
            [
                "text-embeddings-router",
                "--model-id", EMBEDDING_MODEL,
                "--port", str(TEI_PORT),
                "--max-client-batch-size", str(TEI_MAX_BATCH),
                "--auto-truncate",
                "--json-output",
            ],
        )
        self._wait_for_tei()

    def _wait_for_tei(self, timeout: float = 120.0) -> None:
        """Poll TEI socket until it accepts connections."""
        start = time.monotonic()
        while time.monotonic() - start < timeout:
            try:
                with socket.create_connection(("127.0.0.1", TEI_PORT), timeout=1.0):
                    print("TEI server ready", flush=True)
                    return
            except OSError:
                time.sleep(0.5)
        raise TimeoutError(f"TEI did not start within {timeout}s")

    # ── Setup ───────────────────────────────────────────────────────────

    def _init_splitter(self) -> None:
        from llama_index.core.node_parser import TokenTextSplitter
        self._splitter = TokenTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    def _init_chroma(self) -> None:
        import chromadb

        _CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        self._collection = client.get_or_create_collection(CHROMA_COLLECTION)

