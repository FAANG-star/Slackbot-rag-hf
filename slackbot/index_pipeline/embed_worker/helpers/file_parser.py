"""Document and zip file parsing for embedding.

Parses loose files (PDF, DOCX, plaintext) and zip archives into LlamaIndex
Documents, then feeds them through an embed_fn callback in batches.
Returns (chunk_count, chunks) where chunks are ready for ChromaDB upsert.
"""

import io
import zipfile
from pathlib import Path
from typing import Callable

from ...config import EMBED_BATCH_SIZE


class FileParser:

    def __init__(self, embed_fn: Callable[[list], tuple[int, list]]):
        # embed_fn: takes Documents, returns (count, [(id, embedding, text, metadata), ...])
        self._embed_fn = embed_fn

    def embed(self, work: dict) -> tuple[int, list]:
        """Dispatch to the right parser based on work type."""
        if work["type"] == "files":
            return self._embed_files(work["paths"])
        elif work["type"] == "zip_entries":
            return self._embed_zip(work["zip_path"], work["entries"])
        raise ValueError(f"Unknown work type: {work['type']}")

    def _embed_files(self, paths: list[str]) -> tuple[int, list]:
        """Parse each file via SimpleDirectoryReader and embed."""
        from llama_index.core import SimpleDirectoryReader

        total, pending = 0, []
        for path in paths:
            source = Path(path).name
            docs = SimpleDirectoryReader(input_files=[path]).load_data()
            for doc in docs:
                doc.metadata["source"] = source
            count, chunks = self._embed_fn(docs)
            total += count
            pending.extend(chunks)
        return total, pending

    def _embed_zip(self, zip_path: str, entries: list[str]) -> tuple[int, list]:
        """Read assigned zip entries, batch into Documents, and embed."""
        total, pending = 0, []
        with zipfile.ZipFile(zip_path) as zf:
            for batch in self._iter_zip_batches(zf, entries, Path(zip_path).name):
                count, chunks = self._embed_fn(batch)
                total += count
                pending.extend(chunks)
        return total, pending

    def _iter_zip_batches(self, zf: zipfile.ZipFile, entries: list[str], source: str):
        """Yield batches of Documents from zip entries, sized by EMBED_BATCH_SIZE."""
        from llama_index.core import Document

        batch = []
        for name in entries:
            text = self._read_entry(zf, name)
            if text and text.strip():
                batch.append(Document(text=text, metadata={"source": source, "filename": name}))
            if len(batch) >= EMBED_BATCH_SIZE:
                yield batch
                batch = []
        if batch:
            yield batch

    def _read_entry(self, zf: zipfile.ZipFile, name: str) -> str | None:
        """Extract text from a single zip entry. Handles PDF and plaintext."""
        try:
            data = zf.read(name)
            if name.lower().endswith(".pdf"):
                from pypdf import PdfReader
                return "\n".join(p.extract_text() or "" for p in PdfReader(io.BytesIO(data)).pages)
            return data.decode("utf-8", errors="replace")
        except Exception:
            return None
