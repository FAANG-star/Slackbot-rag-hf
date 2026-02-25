"""Router — parse @mention events and dispatch to the correct handler."""

import re

from .index_handler import IndexHandler
from .ml_handler import MlHandler
from .rag_handler import RagHandler


class Router:
    """Parse Slack @mention events and dispatch to the correct handler."""

    def __init__(self, indexer, rag, ml_sb_fn, vol):
        self._index = IndexHandler(indexer, vol)
        self._ml = MlHandler(ml_sb_fn)
        self._rag = RagHandler(rag, vol)

    def handle(self, event: dict, client) -> None:
        channel = event["channel"]
        thread_ts = event.get("thread_ts", event["ts"])
        say = lambda text: client.chat_postMessage(channel=channel, text=text, thread_ts=thread_ts)

        # Strip the @mention tag to get the raw message
        message = re.sub(r"<@[A-Z0-9]+>", "", event.get("text", "")).strip()

        try:
            if event.get("files"):
                self._index.handle(event["files"], channel, thread_ts, say)
            elif message.lower().startswith("hf:"):
                self._ml.handle(message[3:].strip(), thread_ts, say)
            else:
                self._rag.handle(message, thread_ts, channel, say, client)

        except Exception as e:
            print(f"[router] error: {e}", flush=True)
            say(f":x: Error: {e}")
