"""Local LLM — direct vLLM adapter for LlamaIndex."""

import asyncio
from functools import partial
from typing import Any, AsyncGenerator, Generator, Sequence

from llama_index.core.llms import (
    LLM as LlamaIndexLLM,
    LLMMetadata,
    ChatMessage,
    ChatResponse,
    CompletionResponse,
    MessageRole,
)
from llama_index.core.bridge.pydantic import Field, PrivateAttr

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


class VllmAdapter(LlamaIndexLLM):
    """LlamaIndex LLM adapter using vLLM directly — no llama-index-llms-vllm dependency."""

    model_name: str = Field(default=LLM_MODEL)
    max_new_tokens: int = Field(default=4096)
    temperature: float = Field(default=0.7)
    top_p: float = Field(default=0.8)
    context_window: int = Field(default=16384)

    _engine: Any = PrivateAttr()
    _tokenizer: Any = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        from vllm import LLM as _VllmEngine
        from transformers import AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._engine = _VllmEngine(
            model=self.model_name,
            download_dir="/data/hf-cache",
            gpu_memory_utilization=0.75,
            quantization="awq_marlin",
            max_model_len=16384,
            dtype="auto",
            compilation_config={"cache_dir": "/data/vllm-cache"},
        )

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.max_new_tokens,
            model_name=self.model_name,
        )

    def _sampling_params(self):
        from vllm import SamplingParams
        return SamplingParams(
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            presence_penalty=1.5,
        )

    def _messages_to_prompt(self, messages: Sequence[ChatMessage]) -> str:
        dicts = [{"role": m.role.value, "content": m.content} for m in messages]
        return self._tokenizer.apply_chat_template(
            dicts, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )

    def _run(self, prompt: str) -> str:
        outputs = self._engine.generate(prompt, self._sampling_params())
        return outputs[0].outputs[0].text

    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        text = self._run(self._messages_to_prompt(messages))
        return ChatResponse(message=ChatMessage(role=MessageRole.ASSISTANT, content=text))

    def stream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> Generator:
        yield self.chat(messages, **kwargs)

    def complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        return CompletionResponse(text=self._run(prompt))

    def stream_complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> Generator:
        yield self.complete(prompt, **kwargs)

    async def achat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, partial(self.chat, messages, **kwargs))

    async def astream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> AsyncGenerator:
        resp = await self.achat(messages, **kwargs)

        async def _gen():
            yield resp

        return _gen()

    async def acomplete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, partial(self.complete, prompt, **kwargs))

    async def astream_complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> AsyncGenerator:
        resp = await self.acomplete(prompt, **kwargs)

        async def _gen():
            yield resp

        return _gen()


class LLM:
    def __init__(self, model_name: str = LLM_MODEL):
        self.model = VllmAdapter(model_name=model_name)
