"""Command implementations for the RAG agent."""

from __future__ import annotations

import io
import shutil
from contextlib import redirect_stdout
from typing import TYPE_CHECKING

from rag.config import CHROMA_DIR, OUTPUT_DIR

if TYPE_CHECKING:
    from rag.history import HistoryManager
    from rag.indexer import Indexer
    from .container import ServiceContainer


def reload(indexer: Indexer):
    """Reload ChromaDB and manifest after external index changes."""
    import modal

    print("Reloading volume...", flush=True)
    modal.Volume.from_name("sandbox-rag").reload()
    print("Rebuilding index...", flush=True)
    indexer.reload()
    print(f"Index reloaded. {indexer.stats()}", flush=True)


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


def clear(indexer: Indexer, history: HistoryManager, session_id: str):
    indexer.reset()
    shutil.rmtree(CHROMA_DIR, ignore_errors=True)
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    history.clear(session_id)
    print("Index, output files, and conversation history cleared.", flush=True)


def dispatch(cmd: str, container: ServiceContainer) -> str:
    """Route a command string to the appropriate function, return captured output."""
    lower = cmd.strip().lower()
    buf = io.StringIO()
    with redirect_stdout(buf):
        if lower == "reload":
            reload(container.indexer)
        elif lower.startswith("reindex"):
            reindex(container.indexer, "--force" in lower)
        elif lower == "status":
            status(container.indexer)
        elif lower == "clear":
            clear(container.indexer, container.history, cmd)
        else:
            return f"Unknown command: {cmd}"
    return buf.getvalue().strip()
