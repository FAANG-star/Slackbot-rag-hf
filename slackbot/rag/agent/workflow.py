"""AgentWorkflow factory — assembles ReAct agent with search/code/list tools."""

from __future__ import annotations

from typing import TYPE_CHECKING

from llama_index.core.agent.workflow import AgentWorkflow, ReActAgent
from llama_index.core.tools import FunctionTool

from ..config import SYSTEM_PROMPT
from ..llm import LLM
from .tools import execute_python, list_documents, search_documents

if TYPE_CHECKING:
    from ..db import SearchIndex


def create_workflow(search_index: SearchIndex, llm: LLM) -> AgentWorkflow:
    """Create a ReAct agent with search, code execution, and file listing tools."""
    def _search(query: str) -> str:
        return search_documents(query, search_index)

    tools = [
        FunctionTool.from_defaults(
            fn=_search,
            name="search_documents",
            description="Search indexed documents for relevant information.",
        ),
        FunctionTool.from_defaults(
            fn=execute_python,
            name="execute_python",
            description="Execute Python code in a subprocess. "
            "Working directory is /data/rag/. "
            "Pre-installed: pandas, matplotlib, openpyxl, pypdf, python-docx.",
        ),
        FunctionTool.from_defaults(
            fn=list_documents,
            name="list_documents",
            description="List all files in /data/rag/docs/.",
        ),
    ]
    agent = ReActAgent(
        tools=tools,
        llm=llm.model,
        system_prompt=SYSTEM_PROMPT,
        max_iterations=10,
        verbose=True,
    )
    return AgentWorkflow(agents=[agent])
