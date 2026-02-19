"""Parses files and chunks them into LlamaIndex nodes."""

from pathlib import Path

from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter

from ..config import CHUNK_OVERLAP, CHUNK_SIZE, DOCS_DIR


class DocumentParser:
    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        self._splitter = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def parse(self, filenames: list[str], docs_dir: Path = DOCS_DIR):
        """Parse and chunk files. Yields (filename, nodes) pairs."""
        for fname in filenames:
            path = docs_dir / fname
            docs = SimpleDirectoryReader(input_files=[str(path)]).load_data()
            for doc in docs:
                doc.metadata["source"] = fname
            nodes = self._splitter.get_nodes_from_documents(docs)
            yield fname, nodes
