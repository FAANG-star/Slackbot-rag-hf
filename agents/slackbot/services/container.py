"""Service container — singleton lifecycle for all slackbot services."""

from agents.infra.shared import rag_vol
from agents.ml_agent.client import MlClient
from agents.rag_agent.client import RagClient

from .file_manager import FileManager
from .index_client import IndexClient
from .router import MessageRouter


class ServiceContainer:
    """Constructs and holds singleton services for the container lifetime."""

    def __init__(self):
        self.rag = RagClient()
        self.ml = MlClient()
        self.files = FileManager(volume=rag_vol)
        self.indexer = IndexClient(volume=rag_vol)
        self.router = MessageRouter(
            rag=self.rag, ml=self.ml, files=self.files, indexer=self.indexer
        )
