"""AgentWorkflow factory — search + code execution tools, workflow creation."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from llama_index.core.agent.workflow import AgentWorkflow, ReActAgent
from llama_index.core.tools import FunctionTool

from .config import DOCS_DIR, TOP_K
from .llm import LLM, SYSTEM_PROMPT

if TYPE_CHECKING:
    from .indexer import Indexer

OUTPUT_DIR = Path("/data/rag/output")
MAX_ITERATIONS = 10


# ---------------------------------------------------------------------------
# Tools (stateless)
# ---------------------------------------------------------------------------


def _execute_python(code: str) -> str:
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
            cwd=str(OUTPUT_DIR.parent),  # /data/rag — relative paths like docs/x.csv and output/x.png work
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


# ---------------------------------------------------------------------------
# Output file helpers
# ---------------------------------------------------------------------------


def list_output_files() -> list[str]:
    """Return paths of files in the output directory."""
    if not OUTPUT_DIR.exists():
        return []
    return [str(p) for p in sorted(OUTPUT_DIR.iterdir()) if p.is_file()]


def _list_documents() -> str:
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


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------


class SearchStats:
    """Tracks retrieval stats across tool calls within a single query."""

    def __init__(self):
        self.searches: list[tuple[str, int]] = []

    def record(self, query: str, num_chunks: int):
        self.searches.append((query, num_chunks))

    def format(self, elapsed: float = 0.0) -> str:
        parts = []
        for query, count in self.searches:
            parts.append(f"_{query}_ ({count} chunks)")
        if not parts and not elapsed:
            return ""
        timer = f"{elapsed:.1f}s"
        if not parts:
            return f"\n\n> :clock1: {timer}"
        return "\n\n> :mag: " + " | ".join(parts) + f"  :clock1: {timer}"


def create_workflow(indexer: Indexer, llm: LLM) -> tuple[AgentWorkflow, SearchStats]:
    """Create an AgentWorkflow with a ReActAgent + search/code tools."""
    stats = SearchStats()

    def search_documents(query: str) -> str:
        """Search indexed documents for relevant information.

        Use this before answering questions about uploaded files.
        Returns relevant text chunks with source labels.
        """
        if not indexer.has_index():
            return "No documents indexed yet. Ask the user to upload files and run reindex."

        retriever = indexer.index.as_retriever(similarity_top_k=TOP_K)
        nodes = retriever.retrieve(query)
        stats.record(query, len(nodes))
        print(f"[SEARCH] {query!r} -> {len(nodes)} chunks", file=sys.stderr, flush=True)

        if not nodes:
            return "No relevant documents found for this query."

        chunks = []
        for node in nodes:
            source = node.metadata.get("source", "unknown")
            chunks.append(f"[Source: {source}]\n{node.get_content()}")
        return "\n\n---\n\n".join(chunks)

    def execute_python_logged(code: str) -> str:
        # ReAct text format sometimes passes literal \n instead of real newlines
        code = code.replace("\\n", "\n").replace("\\t", "\t")
        print(f"[EXECUTE_PYTHON] code:\n{code}", file=sys.stderr, flush=True)
        result = _execute_python(code)
        print(f"[EXECUTE_PYTHON] output:\n{result}", file=sys.stderr, flush=True)
        return result

    search_tool = FunctionTool.from_defaults(
        fn=search_documents,
        name="search_documents",
        description=(
            "Search indexed documents for relevant information. "
            "Use this before answering questions about uploaded files."
        ),
    )
    code_tool = FunctionTool.from_defaults(
        fn=execute_python_logged,
        name="execute_python",
        description=(
            "Execute Python code in a subprocess. "
            "Working directory is /data/rag/ — use relative paths: docs/file.csv, output/chart.png. "
            "ALWAYS use this to read files, analyze data, or produce charts. "
            "Pre-installed: pandas, matplotlib, openpyxl, pypdf, python-docx."
        ),
    )
    list_tool = FunctionTool.from_defaults(
        fn=_list_documents,
        name="list_documents",
        description=(
            "List all files available in /data/rag/docs/. "
            "Call this first when the user asks about a file, to confirm it exists and get the exact path."
        ),
    )

    react_agent = ReActAgent(
        tools=[search_tool, code_tool, list_tool],
        llm=llm.model,
        system_prompt=SYSTEM_PROMPT,
        verbose=True,
    )
    return AgentWorkflow(agents=[react_agent]), stats
