"""Message routing — unified dispatch for DMs and channel mentions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

from .formatter import say_chunked

if TYPE_CHECKING:
    from agents.ml_agent.client import MlClient
    from agents.rag_agent.client import RagClient

    from .file_manager import FileManager
    from .index_client import IndexClient


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

    def __init__(self, rag: RagClient, ml: MlClient, files: FileManager, indexer: IndexClient):
        self._rag = rag
        self._ml = ml
        self._files = files
        self._indexer = indexer

    def route(self, ctx: MessageContext) -> None:
        """Route a message to the correct handler."""
        lower = ctx.message.strip().lower()
        files_count = len(ctx.files) if ctx.files else 0
        print(f"[router] msg={ctx.message[:60]!r} files={files_count}", flush=True)
        try:
            if ctx.files:
                self._handle_files(ctx)
            elif lower.startswith("reindex"):
                self._handle_reindex(ctx)
            elif lower.startswith("hf:"):
                self._handle_ml(ctx)
            elif lower == "sources":
                ctx.say(self._files.list_sources())
            elif lower.startswith("remove "):
                self._handle_remove(ctx)
            else:
                self._handle_rag_query(ctx)
        except Exception as e:
            print(f"[router] error: {e}", flush=True)
            ctx.say(f":x: Error: {e}")

    def _handle_files(self, ctx: MessageContext):
        print(f"[router] handling {len(ctx.files)} file(s)", flush=True)
        ctx.set_status("downloading files...")
        saved = self._files.download_slack_files(ctx.files)
        if not saved:
            print("[router] no downloadable files found", flush=True)
            ctx.say("No downloadable files found in the shared items.")
            return
        print(f"[router] saved: {saved}", flush=True)
        ctx.say(f"Saved {len(saved)} file(s): {', '.join(saved)}")
        self._do_reindex(ctx, force=False)

    def _handle_reindex(self, ctx: MessageContext):
        force = "--force" in ctx.message.lower()
        self._do_reindex(ctx, force=force)

    def _do_reindex(self, ctx: MessageContext, force: bool):
        print(f"[router] starting parallel reindex (force={force})", flush=True)
        result = self._indexer.reindex(
            force=force,
            on_status=ctx.set_status,
            reload_fn=lambda: self._rag.query("reload", ctx.sandbox_name),
        )
        print(f"[router] reindex done: {result[:80]}", flush=True)
        say_chunked(ctx.say, result)

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
