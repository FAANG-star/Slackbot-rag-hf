"""Stateless tool functions for the ReAct agent."""

import subprocess
import sys

from ..config import DOCS_DIR, OUTPUT_DIR, TOP_K


def search_documents(query: str, search_index) -> str:
    """Search indexed documents for relevant chunks."""
    if not search_index.has_index():
        return "No documents indexed yet. Ask the user to upload files and run reindex."
    retriever = search_index.index.as_retriever(similarity_top_k=TOP_K)
    nodes = retriever.retrieve(query)
    print(f"[SEARCH] {query!r} -> {len(nodes)} chunks", file=sys.stderr, flush=True)
    if not nodes:
        return "No relevant documents found for this query."
    chunks = [f"[Source: {n.metadata.get('source', 'unknown')}]\n{n.get_content()}" for n in nodes]
    return "\n\n---\n\n".join(chunks)


def execute_python(code: str) -> str:
    """Execute Python code for data analysis.

    Pre-installed: pandas, matplotlib, openpyxl, pypdf, python-docx.
    Save output files to /data/rag/output/.
    Input documents are at /data/rag/docs/.
    """
    code = code.replace("\\n", "\n").replace("\\t", "\t")
    print(f"[EXECUTE_PYTHON] code:\n{code}", file=sys.stderr, flush=True)
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
    output = _format_result(result)
    print(f"[EXECUTE_PYTHON] output:\n{output}", file=sys.stderr, flush=True)
    return output


def list_documents() -> str:
    """List all files available in the documents directory (/data/rag/docs/).

    Call this to discover what files are available before analyzing them.
    Returns a newline-separated list of full file paths.
    """
    if not DOCS_DIR.exists():
        return "No documents directory found at /data/rag/docs/"
    paths = [str(f) for f in sorted(DOCS_DIR.rglob("*")) if f.is_file()]
    return "\n".join(paths) if paths else "No files found in /data/rag/docs/"


def list_output_files() -> list[str]:
    """Return paths of files in the output directory."""
    if not OUTPUT_DIR.exists():
        return []
    return [str(p) for p in sorted(OUTPUT_DIR.iterdir()) if p.is_file()]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _format_result(result: subprocess.CompletedProcess) -> str:
    parts = []
    if result.stdout:
        parts.append(result.stdout)
    if result.stderr:
        parts.append(f"[STDERR]\n{result.stderr}")
    if result.returncode != 0:
        parts.append(f"[Exit code: {result.returncode}]")
    return "\n".join(parts).strip() or "[No output]"
