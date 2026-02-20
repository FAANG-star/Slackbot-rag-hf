"""Service container — singleton lifecycle for all slackbot services."""

from agents.slackbot.infra import rag_vol
from agents.index_pipeline.pipeline import IndexPipeline
from .clients.ml_client import MlClient
from .clients.rag_client import RagClient
from .services.documents import Documents
from .services.router import MessageRouter


class ServiceContainer:
    """Constructs and holds singleton services for the container lifetime."""

    def __init__(self):
        self.rag = RagClient()
        self.ml = MlClient()
        self.files = Documents(volume=rag_vol)
        self.indexer = IndexPipeline(volume=rag_vol)
        self.router = MessageRouter(
            rag=self.rag, ml=self.ml, files=self.files, indexer=self.indexer
        )
