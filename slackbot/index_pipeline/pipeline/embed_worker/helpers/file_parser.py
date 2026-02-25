"""Parse files and zip archives into LlamaIndex Documents."""

import io
import zipfile
from pathlib import Path


class FileParser:

    def parse(self, work: dict) -> list:
        """Parse a work unit into Documents ready for embedding."""
        if work["type"] == "files":
            return self._parse_files(work["paths"])
        elif work["type"] == "zip_entries":
            return self._parse_zip(work["zip_path"], work["entries"])
        raise ValueError(f"Unknown work type: {work['type']}")

    def _parse_files(self, paths: list[str]) -> list:
        """Parse loose files (PDF, DOCX, plaintext) via SimpleDirectoryReader.

        Each doc gets source (filename) and fingerprint (mtime:size) metadata
        so Scanner can detect changes on subsequent runs.
        """
        from llama_index.core import SimpleDirectoryReader

        docs = []
        for path in paths:
            p = Path(path)
            fingerprint = _fingerprint(p)
            for doc in SimpleDirectoryReader(input_files=[path]).load_data():
                doc.metadata["source"] = p.name
                doc.metadata["fingerprint"] = fingerprint
                docs.append(doc)
        return docs

    def _parse_zip(self, zip_path: str, entries: list[str]) -> list:
        """Read assigned zip entries into Documents.

        Each entry is extracted as text (PDF pages joined, plaintext decoded).
        All entries share the zip file's source name and fingerprint.
        """
        from llama_index.core import Document

        p = Path(zip_path)
        fingerprint = _fingerprint(p)
        docs = []
        with zipfile.ZipFile(zip_path) as zf:
            for name in entries:
                text = _read_zip_entry(zf, name)
                if text and text.strip():
                    docs.append(Document(
                        text=text,
                        metadata={"source": p.name, "filename": name, "fingerprint": fingerprint},
                    ))
        return docs


def _fingerprint(path: Path) -> str:
    """mtime_ns:size — stored in ChromaDB metadata for incremental indexing."""
    stat = path.stat()
    return f"{stat.st_mtime_ns}:{stat.st_size}"


def _read_zip_entry(zf: zipfile.ZipFile, name: str) -> str | None:
    """Extract text from a zip entry. PDFs via pypdf, everything else as UTF-8."""
    try:
        data = zf.read(name)
        if name.lower().endswith(".pdf"):
            from pypdf import PdfReader
            return "\n".join(p.extract_text() or "" for p in PdfReader(io.BytesIO(data)).pages)
        return data.decode("utf-8", errors="replace")
    except Exception:
        return None
