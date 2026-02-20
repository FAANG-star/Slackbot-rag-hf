"""Parses files and chunks them into LlamaIndex nodes."""

import io
import zipfile
from pathlib import Path

from llama_index.core import Document, SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter

from ..config import CHUNK_OVERLAP, CHUNK_SIZE, DOCS_DIR

ZIP_BATCH_SIZE = 1000  # entries to parse before yielding a batch


class DocumentParser:
    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        self._splitter = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def parse(self, filenames: list[str], docs_dir: Path = DOCS_DIR):
        """Parse and chunk files. Yields (filename, nodes) pairs.

        Zips are streamed in batches to avoid loading all entries into memory.
        """
        for fname in filenames:
            path = docs_dir / fname
            if fname.endswith(".zip"):
                yield from self._parse_zip(fname, path)
            else:
                docs = SimpleDirectoryReader(input_files=[str(path)]).load_data()
                for doc in docs:
                    doc.metadata["source"] = fname
                yield fname, self._splitter.get_nodes_from_documents(docs)

    def _parse_zip(self, fname: str, path: Path):
        """Stream zip entries in batches, yielding (filename, nodes) per batch."""
        from pypdf import PdfReader

        batch = []
        count = 0
        with zipfile.ZipFile(path) as zf:
            for name in zf.namelist():
                if name.endswith("/"):
                    continue
                try:
                    data = zf.read(name)
                    if name.lower().endswith(".pdf"):
                        reader = PdfReader(io.BytesIO(data))
                        text = "\n".join(
                            page.extract_text() or "" for page in reader.pages
                        )
                    else:
                        text = data.decode("utf-8", errors="replace")
                except Exception:
                    continue
                if text.strip():
                    batch.append(Document(text=text, metadata={"source": fname, "filename": name}))
                if len(batch) >= ZIP_BATCH_SIZE:
                    count += len(batch)
                    print(f"[parser] {fname}: chunking batch ({count} entries so far)", flush=True)
                    yield fname, self._splitter.get_nodes_from_documents(batch)
                    batch = []
        if batch:
            count += len(batch)
            print(f"[parser] {fname}: final batch ({count} entries total)", flush=True)
            yield fname, self._splitter.get_nodes_from_documents(batch)
