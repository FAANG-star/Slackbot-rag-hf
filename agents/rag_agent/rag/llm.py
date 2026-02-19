"""Local LLM — HuggingFaceLLM wrapper for generation."""

import torch
from llama_index.llms.huggingface import HuggingFaceLLM
from transformers import BitsAndBytesConfig

from .config import LLM_MODEL

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
    "- Be concise. Cite sources from search results."
)

_bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)


class LLM:
    def __init__(self, model_name: str = LLM_MODEL):
        self.model = HuggingFaceLLM(
            model_name=model_name,
            tokenizer_name=model_name,
            max_new_tokens=4096,
            device_map="auto",
            model_kwargs={"quantization_config": _bnb_config},
            generate_kwargs={"temperature": 0.7, "top_p": 0.9},
        )
