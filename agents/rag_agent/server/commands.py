"""Command implementations for the RAG agent message handler."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from rag.config import CHROMA_DIR

if TYPE_CHECKING:
    from rag.history import HistoryManager
    from rag.indexer import Indexer

OUTPUT_DIR = Path("/data/rag/output")


def reload(indexer: Indexer) -> Indexer:
    """Reload ChromaDB and manifest after external index changes."""
    import modal

    from rag.indexer import Indexer as _Indexer

    print("Reloading volume...", flush=True)
    modal.Volume.from_name("sandbox-rag").reload()
    print("Rebuilding index...", flush=True)
    new_indexer = _Indexer()
    print(f"Index reloaded. {new_indexer.stats()}", flush=True)
    return new_indexer


def reindex(indexer: Indexer, force: bool):
    print("Reading documents from volume...", flush=True)
    total, added, deleted = indexer.build(force=force)
    if total == 0:
        print("No documents found in /data/rag/docs/. Share files via Slack first.", flush=True)
    else:
        mode = "Full rebuild" if force else "Incremental update"
        print(f"{mode}: +{added} chunks added, -{deleted} removed. Total: {total} chunks.", flush=True)


def status(indexer: Indexer):
    print(indexer.stats(), flush=True)


def clear(indexer: Indexer, history: HistoryManager, sandbox_name: str):
    indexer.reset()
    shutil.rmtree(CHROMA_DIR, ignore_errors=True)
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    history.clear(sandbox_name)
    print("Index, output files, and conversation history cleared.", flush=True)
