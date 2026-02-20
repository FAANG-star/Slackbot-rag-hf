"""QueryService — orchestrates memory, workflow, and persistence for a query."""

from __future__ import annotations

import asyncio
import shutil
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

from rag.engine import create_workflow
from rag.tools import list_output_files

if TYPE_CHECKING:
    from rag.history import HistoryManager
    from rag.indexer import Indexer
    from rag.llm import LLM

OUTPUT_DIR = Path("/data/rag/output")


class QueryService:
    """Runs a single RAG query: memory -> workflow -> response -> persist."""

    def __init__(self, llm: LLM, indexer: Indexer, history: HistoryManager):
        self.llm = llm
        self.indexer = indexer
        self._history = history

    def run(self, msg: str, sandbox_name: str):
        """Execute query and print results to stdout."""
        shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

        memory = self._history.get(sandbox_name, llm=self.llm.model)
        workflow, stats = create_workflow(self.indexer, self.llm, memory=memory)

        print("Searching...", file=sys.stderr, flush=True)

        async def _run():
            return await workflow.run(user_msg=msg)

        t0 = time.monotonic()
        response = asyncio.run(_run())
        elapsed = time.monotonic() - t0

        print(str(response) + stats.format(elapsed), flush=True)

        for path in list_output_files():
            print(f"[OUTPUT_FILE:{path}]", flush=True)

        self._history.persist(sandbox_name)
