"""Stateless tool functions for the ReAct agent."""

import subprocess
from pathlib import Path

from .config import DOCS_DIR

OUTPUT_DIR = Path("/data/rag/output")


def execute_python(code: str) -> str:
    """Execute Python code for data analysis.

    Pre-installed: pandas, matplotlib, openpyxl, pypdf, python-docx.
    Save output files to /data/rag/output/.
    Input documents are at /data/rag/docs/.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        result = subprocess.run(
            ["python", "-c", code],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(OUTPUT_DIR.parent),
        )
    except subprocess.TimeoutExpired:
        return "[Error: Code execution timed out after 120 seconds]"

    output = ""
    if result.stdout:
        output += result.stdout
    if result.stderr:
        output += f"\n[STDERR]\n{result.stderr}"
    if result.returncode != 0:
        output += f"\n[Exit code: {result.returncode}]"
    return output.strip() or "[No output]"


def list_output_files() -> list[str]:
    """Return paths of files in the output directory."""
    if not OUTPUT_DIR.exists():
        return []
    return [str(p) for p in sorted(OUTPUT_DIR.iterdir()) if p.is_file()]


def list_documents() -> str:
    """List all files available in the documents directory (/data/rag/docs/).

    Call this to discover what files are available before analyzing them.
    Returns a newline-separated list of full file paths.
    """
    if not DOCS_DIR.exists():
        return "No documents directory found at /data/rag/docs/"
    files = sorted(DOCS_DIR.rglob("*"))
    paths = [str(f) for f in files if f.is_file()]
    if not paths:
        return "No files found in /data/rag/docs/"
    return "\n".join(paths)
