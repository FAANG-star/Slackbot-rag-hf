"""Local LLM — HuggingFaceLLM wrapper for generation."""

import torch
from llama_index.llms.huggingface import HuggingFaceLLM
from transformers import BitsAndBytesConfig

from .config import LLM_MODEL

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
