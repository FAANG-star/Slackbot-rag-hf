"""QueryService — orchestrates memory, workflow, and persistence for a query."""

from __future__ import annotations

import asyncio
import shutil
import time
from typing import TYPE_CHECKING

from rag.config import OUTPUT_DIR
from rag.engine import create_workflow
from rag.tools import list_output_files

if TYPE_CHECKING:
    from rag.history import HistoryManager
    from rag.indexer import Indexer
    from rag.llm import LLM


class QueryService:
    """Runs a single RAG query: memory → workflow → response → persist."""

    def __init__(self, llm: LLM, indexer: Indexer, history: HistoryManager):
        self.llm = llm
        self.indexer = indexer
        self._history = history

    def run(self, msg: str, sandbox_name: str):
        """Execute query and print results to stdout."""
        shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
        memory = self._history.get(sandbox_name, llm=self.llm.model)
        workflow, stats = create_workflow(self.indexer, self.llm, memory=memory)
        response, elapsed = self._execute(workflow, msg)
        self._emit_results(response, stats, elapsed)
        self._history.persist(sandbox_name)

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _execute(self, workflow, msg: str) -> tuple:
        """Run the agent workflow and return (response, elapsed_seconds)."""
        async def _run():
            return await workflow.run(user_msg=msg)
        t0 = time.monotonic()
        response = asyncio.run(_run())
        return response, time.monotonic() - t0

    def _emit_results(self, response, stats, elapsed: float):
        """Print response, search stats, and any output file paths to stdout."""
        print(str(response) + stats.format(elapsed), flush=True)
        for path in list_output_files():
            print(f"[OUTPUT_FILE:{path}]", flush=True)
