"""RAG client constants."""

import re
from dataclasses import dataclass, field

END_TURN = "---END_TURN---"
OUTPUT_FILE_RE = re.compile(r"\[OUTPUT_FILE:(.+?)\]")


@dataclass
class RagResponse:
    text: str
    output_files: list[str] = field(default_factory=list)
