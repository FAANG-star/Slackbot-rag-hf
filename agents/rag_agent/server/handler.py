"""MessageHandler — stdin loop and command dispatch."""

from __future__ import annotations

import json
import sys
import traceback
from typing import TYPE_CHECKING

from . import commands

if TYPE_CHECKING:
    from .container import ServiceContainer

END_TURN = "---END_TURN---"


class MessageHandler:
    def __init__(self, container: ServiceContainer):
        self._container = container

    def run(self):
        """Read JSON messages from stdin, dispatch, and delimit with END_TURN."""
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                self._handle(data["message"], data["sandbox_name"])
            except Exception:
                tb = traceback.format_exc()
                print(tb, file=sys.stderr, flush=True)
                print(f"Error: {tb}", flush=True)
            print(END_TURN, flush=True)

    def _handle(self, user_msg: str, sandbox_name: str):
        """Dispatch a single message to the appropriate command."""
        msg = user_msg.strip()
        lower = msg.lower()
        c = self._container

        if lower == "reload":
            c.indexer = commands.reload(c.indexer)
            c.query_service.indexer = c.indexer
        elif lower.startswith("reindex"):
            commands.reindex(c.indexer, "--force" in lower)
        elif lower == "status":
            commands.status(c.indexer)
        elif lower == "clear":
            commands.clear(c.indexer, c.history, sandbox_name)
        else:
            c.query_service.run(msg, sandbox_name)
