"""Message routing — unified dispatch for DMs and channel mentions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

from .formatter import say_chunked

if TYPE_CHECKING:
    from agents.ml_agent.client import MlClient
    from agents.rag_agent.client import RagClient

    from .file_manager import FileManager


@dataclass
class MessageContext:
    """Abstracts the difference between DM and channel message APIs."""

    message: str
    sandbox_name: str
    thread_ts: str
    say: Callable[[str], None]
    set_status: Callable[[str], None]
    files: list[dict] | None = None
    client: Any = None
    channel: str | None = None


class MessageRouter:
    """Dispatches messages to the appropriate agent or file operation."""

    def __init__(self, rag: RagClient, ml: MlClient, files: FileManager):
        self._rag = rag
        self._ml = ml
        self._files = files

    def route(self, ctx: MessageContext) -> None:
        """Route a message to the correct handler."""
        lower = ctx.message.strip().lower()
        try:
            if ctx.files:
                self._handle_files(ctx)
            elif lower.startswith("hf:"):
                self._handle_ml(ctx)
            elif lower == "sources":
                ctx.say(self._files.list_sources())
            elif lower.startswith("remove "):
                self._handle_remove(ctx)
            else:
                self._handle_rag_query(ctx)
        except Exception as e:
            ctx.say(f":x: Error: {e}")

    def _handle_files(self, ctx: MessageContext):
        ctx.set_status("downloading files...")
        saved = self._files.download_slack_files(ctx.files)
        if not saved:
            ctx.say("No downloadable files found in the shared items.")
            return
        ctx.say(f"Saved {len(saved)} file(s): {', '.join(saved)}")
        resp = self._rag.query("reindex", ctx.sandbox_name, on_status=ctx.set_status)
        say_chunked(ctx.say, resp.text)

    def _handle_ml(self, ctx: MessageContext):
        ctx.set_status("thinking...")
        msg = ctx.message[3:].strip()
        response = self._ml.run(msg, ctx.sandbox_name)
        say_chunked(ctx.say, response)

    def _handle_remove(self, ctx: MessageContext):
        filename = ctx.message[7:].strip()
        if not filename:
            ctx.say("Usage: `remove <filename>`")
            return
        ctx.set_status("removing file...")
        ctx.say(self._files.remove_source(filename))

    def _handle_rag_query(self, ctx: MessageContext):
        ctx.set_status("Searching...")
        resp = self._rag.query(
            ctx.message, ctx.sandbox_name, on_status=ctx.set_status
        )
        say_chunked(ctx.say, resp.text)
        if resp.output_files and ctx.client and ctx.channel:
            self._files.upload_output_files(
                ctx.client, ctx.channel, ctx.thread_ts, resp.output_files
            )
