"""Slack message formatting utilities."""

from typing import Callable

SLACK_MSG_LIMIT = 3900


def chunk_message(text: str, limit: int = SLACK_MSG_LIMIT) -> list[str]:
    """Split text into chunks respecting Slack's message limit."""
    if len(text) <= limit:
        return [text]

    chunks = []
    while text:
        if len(text) <= limit:
            chunks.append(text)
            break
        split_at = text.rfind("\n", 0, limit)
        if split_at == -1:
            split_at = limit
        chunks.append(text[:split_at])
        text = text[split_at:].lstrip("\n")
    return chunks


def say_chunked(say: Callable[[str], None], text: str) -> None:
    """Post text to Slack, splitting into chunks. Posts fallback if empty."""
    if not text:
        say("Agent returned no output.")
        return
    for chunk in chunk_message(text):
        if chunk.strip():
            say(chunk)
