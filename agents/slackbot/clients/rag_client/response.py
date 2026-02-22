"""Stdout response parsing."""

from .config import END_TURN, OUTPUT_FILE_RE, RagResponse


def parse_response(stdout) -> RagResponse:
    """Read stdout lines until END_TURN, extract output file markers."""
    lines = []
    for line in stdout:
        print(f"[RAG out] {line.rstrip()}", flush=True)
        if END_TURN in line:
            prefix = line[:line.index(END_TURN)].strip()
            if prefix:
                lines.append(prefix)
            break
        lines.append(line)

    text = "\n".join(lines).strip()
    output_files = OUTPUT_FILE_RE.findall(text)
    display_text = OUTPUT_FILE_RE.sub("", text).strip()
    return RagResponse(text=display_text, output_files=output_files)
