"""Local LLM — vLLM wrapper for generation."""

import asyncio
from functools import partial
from typing import Any, Sequence

from llama_index.llms.vllm import Vllm
from llama_index.core.llms import ChatMessage, ChatResponse
from transformers import AutoTokenizer

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


class _AsyncVllm(Vllm):
    """Vllm with async methods via thread executor (base Vllm only implements sync)."""

    async def achat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, partial(self.chat, messages, **kwargs))

    async def astream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any):
        resp = await self.achat(messages, **kwargs)

        async def _gen():
            yield resp

        return _gen()


class LLM:
    def __init__(self, model_name: str = LLM_MODEL):
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        def messages_to_prompt(messages):
            dicts = [{"role": m.role.value, "content": m.content} for m in messages]
            return tokenizer.apply_chat_template(
                dicts, tokenize=False, add_generation_prompt=True
            )

        def completion_to_prompt(completion):
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": completion}],
                tokenize=False,
                add_generation_prompt=True,
            )

        self.model = _AsyncVllm(
            model=model_name,
            max_new_tokens=4096,
            temperature=0.7,
            top_p=0.9,
            dtype="auto",
            download_dir="/data/hf-cache",
            vllm_kwargs={
                "gpu_memory_utilization": 0.85,
                "quantization": "awq_marlin",
                "max_model_len": 8192,
            },
            messages_to_prompt=messages_to_prompt,
            completion_to_prompt=completion_to_prompt,
        )
