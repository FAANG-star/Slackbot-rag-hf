"""Service container — singleton lifecycle for all slackbot services."""

from agents.slackbot.shared import rag_vol
from .ml_client import MlClient
from .rag_client import RagClient

from .file_manager import FileManager
from agents.rag_agent.indexer_workers.pipeline import IndexPipeline
from .router import MessageRouter


class ServiceContainer:
    """Constructs and holds singleton services for the container lifetime."""

    def __init__(self):
        self.rag = RagClient()
        self.ml = MlClient()
        self.files = FileManager(volume=rag_vol)
        self.indexer = IndexPipeline(volume=rag_vol)
        self.router = MessageRouter(
            rag=self.rag, ml=self.ml, files=self.files, indexer=self.indexer
        )
