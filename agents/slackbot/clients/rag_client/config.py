"""RAG client types."""

from dataclasses import dataclass, field


@dataclass
class RagResponse:
    text: str
    output_files: list[str] = field(default_factory=list)
