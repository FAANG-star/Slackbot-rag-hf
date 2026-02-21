"""QueryService — orchestrates memory, workflow, and persistence for a query."""

from __future__ import annotations

import asyncio
import re
import shutil
from typing import TYPE_CHECKING

from rag.config import OUTPUT_DIR
from rag.engine import create_workflow
from rag.tools import list_output_files

if TYPE_CHECKING:
    from rag.history import HistoryManager
    from rag.indexer import Indexer
    from rag.llm import LLM

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


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
        workflow, _ = create_workflow(self.indexer, self.llm, memory=memory)
        response = self._execute(workflow, msg)
        self._emit_results(response)
        self._history.persist(sandbox_name)

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _execute(self, workflow, msg: str):
        """Run the agent workflow and return the response."""
        async def _run():
            return await workflow.run(user_msg=msg)
        return asyncio.run(_run())

    def _emit_results(self, response):
        """Print response and any output file paths to stdout."""
        text = _THINK_RE.sub("", str(response)).strip()
        print(text, flush=True)
        for path in list_output_files():
            print(f"[OUTPUT_FILE:{path}]", flush=True)
