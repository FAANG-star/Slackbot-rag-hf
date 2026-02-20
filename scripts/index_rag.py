"""Parallel RAG indexer — CLI wrapper around shared GPU worker functions.

Usage:
    modal run scripts/index_rag.py                  # index all files
    modal run scripts/index_rag.py --force          # wipe + reindex
    modal run scripts/index_rag.py::finalize_only   # merge existing pkls
"""

import math

import modal

from agents.infra.shared import app
from agents.rag_agent.indexer_workers import EmbedWorker, N_WORKERS, finalize_index

modal.enable_output()


@app.function(
    image=modal.Image.debian_slim(python_version="3.12"),
    volumes={"/data": modal.Volume.from_name("sandbox-rag")},
    timeout=10 * 60,
)
def list_docs() -> list[str]:
    from pathlib import Path

    docs_dir = Path("/data/rag/docs")
    if not docs_dir.exists():
        return []
    return [str(p) for p in sorted(docs_dir.iterdir()) if p.is_file()]


@app.local_entrypoint()
def main(force: bool = False):
    print("Listing docs...", flush=True)
    all_files = list_docs.remote()
    if not all_files:
        print("No files found in /data/rag/docs/.")
        return

    print(f"Found {len(all_files)} file(s).", flush=True)
    chunk_size = math.ceil(len(all_files) / N_WORKERS)
    batches = [all_files[i : i + chunk_size] for i in range(0, len(all_files), chunk_size)]
    worker_ids = list(range(len(batches)))
    print(f"Spawning {len(batches)} workers, ~{chunk_size} file(s) each...", flush=True)

    total_chunks = 0
    failed = 0
    for result in EmbedWorker().embed.map(batches, worker_ids, return_exceptions=True):
        if isinstance(result, Exception):
            failed += 1
            print(f"  WORKER FAILED: {result}", flush=True)
        else:
            wid, count = result
            total_chunks += count
            print(f"  worker-{wid}: {count:,} chunks", flush=True)

    status = f"{total_chunks:,} chunks"
    if failed:
        status += f", {failed} worker(s) failed"
    print(f"\nAll workers done — {status}. Writing to ChromaDB...", flush=True)

    finalize_index.remote(force)


@app.local_entrypoint()
def finalize_only(force: bool = False):
    """Merge existing pkl files into ChromaDB without re-embedding."""
    finalize_index.remote(force)
