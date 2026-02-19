"""Conversation memory — ChatMemoryBuffer with JSON persistence for crash recovery."""

import json
from collections import OrderedDict
from pathlib import Path

from llama_index.core.llms import ChatMessage
from llama_index.core.memory import ChatMemoryBuffer

from .config import HISTORY_DIR, MAX_HISTORY_TURNS, MEMORY_TOKEN_LIMIT

_MAX_CONVERSATIONS = 50
_memories: OrderedDict[str, ChatMemoryBuffer] = OrderedDict()


def _json_path(sandbox_name: str) -> Path:
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    return HISTORY_DIR / f"{sandbox_name}.json"


def _load_json_history(sandbox_name: str) -> list[dict]:
    path = _json_path(sandbox_name)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, ValueError):
        return []
    return data[-(MAX_HISTORY_TURNS * 2) :]


def _save_json_history(sandbox_name: str, messages: list[ChatMessage]):
    capped = messages[-(MAX_HISTORY_TURNS * 2) :]
    data = [{"role": msg.role.value, "content": msg.content} for msg in capped]
    _json_path(sandbox_name).write_text(json.dumps(data, indent=2))


def _delete_json_history(sandbox_name: str):
    path = _json_path(sandbox_name)
    if path.exists():
        path.unlink()


def get_memory(sandbox_name: str, llm=None) -> ChatMemoryBuffer:
    """Get or create memory for a sandbox. Loads from JSON on first access."""
    if sandbox_name in _memories:
        _memories.move_to_end(sandbox_name)
        return _memories[sandbox_name]

    kwargs = {"token_limit": MEMORY_TOKEN_LIMIT}
    if llm is not None:
        kwargs["llm"] = llm
    memory = ChatMemoryBuffer.from_defaults(**kwargs)
    for msg in _load_json_history(sandbox_name):
        memory.put(ChatMessage(role=msg["role"], content=msg["content"]))

    _memories[sandbox_name] = memory
    while len(_memories) > _MAX_CONVERSATIONS:
        _memories.popitem(last=False)
    return memory


def persist_memory(sandbox_name: str):
    """Save current memory to JSON for crash recovery."""
    memory = _memories.get(sandbox_name)
    if memory:
        _save_json_history(sandbox_name, memory.get_all())


def clear_memory(sandbox_name: str):
    """Clear in-memory + persisted history."""
    _memories.pop(sandbox_name, None)
    _delete_json_history(sandbox_name)
