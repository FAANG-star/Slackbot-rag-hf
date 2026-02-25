"""Query execution and response parsing helpers."""

import re
import shutil

from ..config import OUTPUT_DIR
from .tools import list_output_files


async def run_query(message, *, llm, search_index):
    """Execute a RAG workflow and return the response."""
    from .workflow import create_workflow

    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    workflow = create_workflow(search_index, llm)
    return await workflow.run(user_msg=message)


def parse_response(response) -> tuple[str, list[str]]:
    """Strip think tags and collect output files."""
    text = re.sub(r"<think>.*?</think>", "", str(response), flags=re.DOTALL).strip()
    output_files = list_output_files()
    return text, output_files
