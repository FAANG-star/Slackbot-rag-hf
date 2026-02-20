"""Parallel RAG indexer — N GPU workers, each embedding multiple zips.

The model loads once per worker and is reused across all its zips.
Each zip gets its own .pkl in /data/rag/pending/ (idempotent — safe to re-run).
finalize() merges all pkls into ChromaDB and writes the manifest.

Usage:
    modal run scripts/index_rag.py::main                  # index all zips
    modal run scripts/index_rag.py::main --force          # wipe + reindex
    modal run scripts/index_rag.py::finalize_only         # merge existing pkls
"""

import math

import modal

modal.enable_output()

app = modal.App("ml-agent")
rag_vol = modal.Volume.from_name("sandbox-rag")

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

EMBED_BATCH_SIZE = 256  # emails per GPU forward pass
N_WORKERS = 10          # GPU containers (model loads once each)


@app.function(image=_index_image, volumes={"/data": rag_vol}, timeout=10 * 60)
def list_docs() -> list[str]:
    from pathlib import Path
    docs_dir = Path("/data/rag/docs")
    if not docs_dir.exists():
        return []
    return [str(p) for p in sorted(docs_dir.iterdir()) if p.is_file()]


@app.function(image=_index_image, volumes={"/data": rag_vol}, timeout=10 * 60)
def list_pending() -> list[str]:
    """Return stems of pkl files already in /data/rag/pending/."""
    from pathlib import Path
    pending_dir = Path("/data/rag/pending")
    if not pending_dir.exists():
        return []
    return [p.stem for p in pending_dir.glob("*.pkl")]


@app.function(
    image=_index_image,
    volumes={"/data": rag_vol},
    gpu="A10G",
    timeout=60 * 60,
    env={"HF_HOME": "/data/hf-cache"},
)
def embed_zips(zip_paths: list[str]) -> list[tuple[str, int]]:
    """Embed multiple zips sequentially. Model loads once per container."""
    import os
    import pickle
    import zipfile
    from pathlib import Path
    from llama_index.core import Document
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-large-en-v1.5",
        device="cuda",
        normalize=True,
        embed_batch_size=EMBED_BATCH_SIZE,
    )
    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=64)
    pending_dir = Path("/data/rag/pending")
    pending_dir.mkdir(parents=True, exist_ok=True)

    def _embed_batch(docs, records):
        nodes = splitter.get_nodes_from_documents(docs)
        if not nodes:
            return
        texts = [n.get_content() for n in nodes]
        embeddings = embed_model.get_text_embedding_batch(texts, show_progress=False)
        records.extend(
            {"id": n.node_id, "embedding": emb, "document": txt, "metadata": n.metadata}
            for n, emb, txt in zip(nodes, embeddings, texts)
        )

    results = []
    for zip_path in zip_paths:
        source = os.path.basename(zip_path)
        pkl_path = pending_dir / f"{source}.pkl"

        if pkl_path.exists():
            with open(pkl_path, "rb") as f:
                existing = pickle.load(f)
            print(f"  {source}: already done — {len(existing):,} chunks (skipping)", flush=True)
            results.append((source, len(existing)))
            continue

        records = []
        with zipfile.ZipFile(zip_path) as zf:
            names = [n for n in zf.namelist() if not n.endswith("/")]
            batch = []
            for i, name in enumerate(names):
                try:
                    text = zf.read(name).decode("utf-8", errors="replace")
                except Exception:
                    continue
                batch.append(Document(text=text, metadata={"source": source, "filename": name}))
                if len(batch) >= EMBED_BATCH_SIZE:
                    _embed_batch(batch, records)
                    batch = []
                    print(f"  {source}: {i + 1}/{len(names)} emails, {len(records):,} chunks", flush=True)
            if batch:
                _embed_batch(batch, records)

        with open(pkl_path, "wb") as f:
            pickle.dump(records, f)
        rag_vol.commit()

        print(f"  {source}: done — {len(records):,} chunks", flush=True)
        results.append((source, len(records)))

    return results


@app.function(image=_index_image, volumes={"/data": rag_vol}, timeout=60 * 60)
def finalize(force: bool = False):
    """Merge all pending pkl files into ChromaDB and write manifest."""
    import json
    import pickle
    import shutil
    import chromadb
    from pathlib import Path

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

    pkl_files = sorted(PENDING_DIR.glob("*.pkl"))
    print(f"Merging {len(pkl_files)} pkls into ChromaDB...", flush=True)

    total = 0
    UPSERT_BATCH = 5_000
    for pkl_path in pkl_files:
        with open(pkl_path, "rb") as f:
            records = pickle.load(f)
        for i in range(0, len(records), UPSERT_BATCH):
            chunk = records[i:i + UPSERT_BATCH]
            collection.upsert(
                ids=[r["id"] for r in chunk],
                embeddings=[r["embedding"] for r in chunk],
                documents=[r["document"] for r in chunk],
                metadatas=[r["metadata"] for r in chunk],
            )
        total += len(records)
        print(f"  {pkl_path.stem}: {len(records):,} chunks (total: {total:,})", flush=True)

    manifest = {}
    for path in sorted(DOCS_DIR.iterdir()):
        if path.is_file():
            stat = path.stat()
            manifest[path.name] = f"{stat.st_mtime_ns}:{stat.st_size}"
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))

    shutil.rmtree(PENDING_DIR)
    rag_vol.commit()
    print(f"Done — {total:,} chunks from {len(manifest)} files.", flush=True)


@app.local_entrypoint()
def main(force: bool = False):
    import os

    print("Listing docs...", flush=True)
    all_files = list_docs.remote()
    if not all_files:
        print("No files found in /data/rag/docs/.")
        return

    done_stems = set(list_pending.remote())
    pending = [p for p in all_files if os.path.splitext(os.path.basename(p))[0] not in done_stems]

    if done_stems:
        print(f"  {len(done_stems)} zips already embedded, {len(pending)} remaining.", flush=True)

    if pending:
        chunk_size = math.ceil(len(pending) / N_WORKERS)
        batches = [pending[i:i + chunk_size] for i in range(0, len(pending), chunk_size)]
        print(f"Spawning {len(batches)} workers, ~{chunk_size} zips each...", flush=True)

        completed = 0
        failed_batches = 0
        total_chunks = 0
        for result in embed_zips.map(batches, return_exceptions=True):
            if isinstance(result, Exception):
                failed_batches += 1
                print(f"  BATCH FAILED: {result}", flush=True)
            else:
                for source, count in result:
                    completed += 1
                    total_chunks += count
                    print(f"  [{completed}/{len(pending)}] {source}: {count:,} chunks", flush=True)

        status = f"{total_chunks:,} new chunks"
        if failed_batches:
            status += f", {failed_batches} batch(es) failed (will retry on next run)"
        print(f"\nAll workers done — {status}. Writing to ChromaDB...", flush=True)
    else:
        print("All zips already embedded. Writing to ChromaDB...", flush=True)

    finalize.remote(force)


@app.local_entrypoint()
def finalize_only(force: bool = False):
    """Merge existing pkl files into ChromaDB without re-embedding."""
    finalize.remote(force)
