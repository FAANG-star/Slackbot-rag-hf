"""AgentWorkflow factory — search + code execution tools, workflow creation."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from llama_index.core.agent.workflow import AgentWorkflow, ReActAgent
from llama_index.core.tools import FunctionTool

from .config import TOP_K
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
            cwd=str(OUTPUT_DIR),
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


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------


def create_workflow(indexer: Indexer, llm: LLM) -> AgentWorkflow:
    """Create an AgentWorkflow with a ReActAgent + search/code tools."""

    def search_documents(query: str) -> str:
        """Search indexed documents for relevant information.

        Use this before answering questions about uploaded files.
        Returns relevant text chunks with source labels.
        """
        if not indexer.has_index():
            return "No documents indexed yet. Ask the user to upload files and run reindex."

        print(f"[SEARCH] Querying index: {query!r}", file=sys.stderr, flush=True)
        retriever = indexer.index.as_retriever(similarity_top_k=TOP_K)
        nodes = retriever.retrieve(query)
        print(f"[SEARCH] Retrieved {len(nodes)} chunks", file=sys.stderr, flush=True)

        if not nodes:
            return "No relevant documents found for this query."

        chunks = []
        for node in nodes:
            source = node.metadata.get("source", "unknown")
            chunks.append(f"[Source: {source}]\n{node.get_content()}")
        return "\n\n---\n\n".join(chunks)

    search_tool = FunctionTool.from_defaults(
        fn=search_documents,
        name="search_documents",
        description=(
            "Search indexed documents for relevant information. "
            "Use this before answering questions about uploaded files."
        ),
    )
    code_tool = FunctionTool.from_defaults(
        fn=_execute_python,
        name="execute_python",
        description=(
            "Execute Python code for data analysis. "
            "Pre-installed: pandas, matplotlib, openpyxl, pypdf, python-docx. "
            "Save output files to /data/rag/output/. "
            "Input documents are at /data/rag/docs/."
        ),
    )

    react_agent = ReActAgent(
        tools=[search_tool, code_tool],
        llm=llm.model,
        system_prompt=SYSTEM_PROMPT,
        verbose=False,
    )
    return AgentWorkflow(agents=[react_agent])
