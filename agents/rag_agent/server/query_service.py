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

OUTPUT_FILE_RE = re.compile(r"\[OUTPUT_FILE:(.+?)\]")


class QueryService:
    """Runs a single RAG query: memory → workflow → response → persist."""

    def __init__(self, llm: LLM, indexer: Indexer, history: HistoryManager):
        self.llm = llm
        self.indexer = indexer
        self._history = history

    def run(self, msg: str, session_id: str) -> tuple[str, list[str]]:
        """Execute query and return (text, output_files)."""
        shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
        memory = self._history.get(session_id, llm=self.llm.model)
        workflow, _ = create_workflow(self.indexer, self.llm, memory=memory)
        response = self._execute(workflow, msg)
        self._history.persist(session_id)
        return self._parse_response(response)

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _execute(self, workflow, msg: str):
        async def _run():
            return await workflow.run(user_msg=msg)
        return asyncio.run(_run())

    def _parse_response(self, response) -> tuple[str, list[str]]:
        text = _THINK_RE.sub("", str(response)).strip()
        output_files = [str(p) for p in list_output_files()]
        display_text = OUTPUT_FILE_RE.sub("", text).strip()
        return display_text, output_files
