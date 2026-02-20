"""Parses files and chunks them into LlamaIndex nodes."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from llama_index.core import Document, SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter

from ..config import CHUNK_OVERLAP, CHUNK_SIZE, DOCS_DIR


class DocumentParser:
    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        self._splitter = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def parse(self, filenames: list[str], docs_dir: Path = DOCS_DIR):
        """Parse and chunk files. Yields (filename, nodes) pairs.

        File reads are parallelised with ThreadPoolExecutor (I/O-bound).
        """
        def _parse_one(fname):
            import zipfile
            path = docs_dir / fname
            if fname.endswith(".zip"):
                import io
                from pypdf import PdfReader
                docs = []
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
                            docs.append(Document(text=text, metadata={"source": fname, "filename": name}))
            else:
                docs = SimpleDirectoryReader(input_files=[str(path)]).load_data()
                for doc in docs:
                    doc.metadata["source"] = fname
            return fname, self._splitter.get_nodes_from_documents(docs)

        with ThreadPoolExecutor(max_workers=32) as ex:
            yield from ex.map(_parse_one, filenames)
