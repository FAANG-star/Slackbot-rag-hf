"""AgentWorkflow factory — assembles ReAct agent with search/code/list tools."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from llama_index.core.agent.workflow import AgentWorkflow, ReActAgent
from llama_index.core.tools import FunctionTool

from .config import TOP_K
from .llm import LLM
from .search_stats import SearchStats
from .tools import execute_python, list_documents

if TYPE_CHECKING:
    from .indexer import Indexer

SYSTEM_PROMPT = (
    "You are a document assistant with three tools.\n\n"
    "**list_documents()** — lists all files in /data/rag/docs/. "
    "Call this first when the user mentions a file, to confirm it exists and get the exact path.\n\n"
    "**search_documents(query)** — full-text search over indexed documents. "
    "Use for questions about document content, concepts, or facts.\n\n"
    "**execute_python(code)** — runs Python in a subprocess and returns stdout. "
    "Use for data analysis, file reading, chart generation, or any computation. "
    "Save charts/outputs to /data/rag/output/. "
    "Pre-installed: pandas, matplotlib, openpyxl, pypdf, python-docx.\n\n"
    "Rules:\n"
    "- When the user mentions a file: call list_documents first, then execute_python with the exact path.\n"
    "- For document questions: call search_documents first, then answer from results.\n"
    "- Never guess file contents or paths — use tools to check.\n"
    "- Be concise. Cite sources from search results.\n"
    "- Do not use emojis in responses."
)


def create_workflow(
    indexer: Indexer, llm: LLM, memory=None, max_iterations: int = 10
) -> tuple[AgentWorkflow, SearchStats]:
    """Create an AgentWorkflow with a ReActAgent + search/code/list tools."""
    stats = SearchStats()
    tools = [
        _search_tool(indexer, stats),
        _code_tool(),
        _list_tool(),
    ]
    react_agent = ReActAgent(
        tools=tools,
        llm=llm.model,
        system_prompt=SYSTEM_PROMPT,
        memory=memory,
        max_iterations=max_iterations,
        verbose=True,
    )
    return AgentWorkflow(agents=[react_agent]), stats


# ── Tool builders ─────────────────────────────────────────────────────────────

def _search_tool(indexer: Indexer, stats: SearchStats) -> FunctionTool:
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
        chunks = [f"[Source: {n.metadata.get('source', 'unknown')}]\n{n.get_content()}" for n in nodes]
        return "\n\n---\n\n".join(chunks)

    return FunctionTool.from_defaults(
        fn=search_documents,
        name="search_documents",
        description="Search indexed documents for relevant information. "
        "Use this before answering questions about uploaded files.",
    )


def _code_tool() -> FunctionTool:
    def execute_python_logged(code: str) -> str:
        code = code.replace("\\n", "\n").replace("\\t", "\t")
        print(f"[EXECUTE_PYTHON] code:\n{code}", file=sys.stderr, flush=True)
        result = execute_python(code)
        print(f"[EXECUTE_PYTHON] output:\n{result}", file=sys.stderr, flush=True)
        return result

    return FunctionTool.from_defaults(
        fn=execute_python_logged,
        name="execute_python",
        description="Execute Python code in a subprocess. "
        "Working directory is /data/rag/ — use relative paths: docs/file.csv, output/chart.png. "
        "ALWAYS use this to read files, analyze data, or produce charts. "
        "Pre-installed: pandas, matplotlib, openpyxl, pypdf, python-docx.",
    )


def _list_tool() -> FunctionTool:
    return FunctionTool.from_defaults(
        fn=list_documents,
        name="list_documents",
        description="List all files available in /data/rag/docs/. "
        "Call this first when the user asks about a file, to confirm it exists and get the exact path.",
    )
