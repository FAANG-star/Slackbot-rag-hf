"""File parsing utilities — reads files into LlamaIndex Documents."""

import io
import zipfile
from collections.abc import Iterator

from llama_index.core import Document, SimpleDirectoryReader
from pypdf import PdfReader


def parse_file(path: str, source: str) -> list[Document]:
    """Load a regular file and tag with source metadata."""
    docs = SimpleDirectoryReader(input_files=[path]).load_data()
    for doc in docs:
        doc.metadata["source"] = source
    return docs


def parse_zip_batched(path: str, source: str, batch_size: int) -> Iterator[list[Document]]:
    """Yield batches of Documents from a zip archive (streaming, memory-safe)."""
    with zipfile.ZipFile(path) as zf:
        names = [n for n in zf.namelist() if not n.endswith("/")]
        batch: list[Document] = []
        for name in names:
            text = _read_zip_entry(zf, name)
            if not text or not text.strip():
                continue
            batch.append(Document(text=text, metadata={"source": source, "filename": name}))
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch


def read_zip_entries(
    path: str, entry_names: list[str], source: str, batch_size: int
) -> Iterator[list[Document]]:
    """Yield batches of Documents from a specific subset of zip entries."""
    with zipfile.ZipFile(path) as zf:
        batch: list[Document] = []
        for name in entry_names:
            text = _read_zip_entry(zf, name)
            if not text or not text.strip():
                continue
            batch.append(Document(text=text, metadata={"source": source, "filename": name}))
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch


def _read_zip_entry(zf: zipfile.ZipFile, name: str) -> str | None:
    """Extract a single zip entry as text (supports PDF and plaintext)."""
    try:
        data = zf.read(name)
        if name.lower().endswith(".pdf"):
            reader = PdfReader(io.BytesIO(data))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        return data.decode("utf-8", errors="replace")
    except Exception:
        return None
