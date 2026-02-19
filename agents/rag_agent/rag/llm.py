"""Local LLM — HuggingFaceLLM singleton for generation."""

import torch
from llama_index.llms.huggingface import HuggingFaceLLM

from .config import LLM_MODEL

_llm = None

SYSTEM_PROMPT = (
    "You are a helpful assistant. You can answer questions and analyze documents.\n\n"
    "You have two tools available:\n"
    "1. **search_documents** — Search indexed documents for relevant information. "
    "Always use this before answering questions about uploaded files.\n"
    "2. **execute_python** — Execute Python code for data analysis. "
    "Pre-installed libraries: pandas, openpyxl, matplotlib, pypdf, python-docx, json, csv.\n"
    "You can install additional packages with: subprocess.run(['pip', 'install', 'package_name'])\n\n"
    "To save output files, write them to the /data/rag/output/ directory. "
    "For example: df.to_csv('/data/rag/output/results.csv')\n\n"
    "Input documents are available at /data/rag/docs/ for direct file access.\n\n"
    "If the context doesn't contain relevant information, say so honestly. Be concise but thorough."
)


def get_llm() -> HuggingFaceLLM:
    """Load or return cached HuggingFaceLLM (runs on GPU)."""
    global _llm
    if _llm is None:
        _llm = HuggingFaceLLM(
            model_name=LLM_MODEL,
            tokenizer_name=LLM_MODEL,
            device_map="auto",
            model_kwargs={"torch_dtype": torch.float16},
            generate_kwargs={"max_new_tokens": 4096, "temperature": 0.7, "top_p": 0.9},
        )
    return _llm
