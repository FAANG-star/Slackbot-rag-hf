"""ServiceContainer — constructs and owns all RAG agent services."""

import sys

from rag.history import HistoryManager
from rag.indexer import Indexer
from rag.llm import LLM

from .query_service import QueryService


class ServiceContainer:
    """Singleton lifecycle for all RAG agent services."""

    def __init__(self):
        print("Loading LLM...", file=sys.stderr, flush=True)
        self.llm = LLM()

        print("Loading indexer...", file=sys.stderr, flush=True)
        self.indexer = Indexer()

        self.history = HistoryManager()
        self.query_service = QueryService(
            llm=self.llm, indexer=self.indexer, history=self.history
        )
