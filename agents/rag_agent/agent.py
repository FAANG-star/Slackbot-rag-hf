#!/usr/bin/env python3
"""RAG agent server — long-running process, models stay loaded in VRAM.

Reads JSON messages from stdin, writes responses to stdout.
Each response ends with END_TURN sentinel.

Commands:
    {"message": "reindex", "sandbox_name": "..."}
    {"message": "reindex --force", "sandbox_name": "..."}
    {"message": "status", "sandbox_name": "..."}
    {"message": "clear", "sandbox_name": "..."}
    {"message": "<any question>", "sandbox_name": "..."}
"""

import json
import shutil
import sys
import traceback
from pathlib import Path

from rag.config import CHROMA_DIR
from rag.engine import create_agent, list_output_files
from rag.history import clear_memory, get_memory, persist_memory
from rag.indexer import Indexer
from rag.llm import get_llm

END_TURN = "---END_TURN---"
OUTPUT_DIR = Path("/data/rag/output")


class MessageHandler:
    def __init__(self, indexer: Indexer):
        self.indexer = indexer

    def run(self):
        """Read JSON messages from stdin, dispatch, and delimit with END_TURN."""
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                self.handle(data["message"], data["sandbox_name"])
            except Exception:
                traceback.print_exc()
                print("Error processing message", flush=True)
            print(END_TURN, flush=True)

    def handle(self, user_msg: str, sandbox_name: str):
        """Process a single message. All output goes to stdout."""
        msg = user_msg.strip()
        lower = msg.lower()

        if lower.startswith("reindex"):
            self._reindex(lower)
        elif lower == "status":
            self._status()
        elif lower == "clear":
            self._clear(sandbox_name)
        else:
            self._query(msg, sandbox_name)

    def _reindex(self, lower: str):
        force = "--force" in lower
        print("Reading documents from volume...", flush=True)
        total, added, deleted = self.indexer.build(force=force)
        if total == 0:
            print("No documents found in /data/rag/docs/. Share files via Slack first.", flush=True)
        else:
            mode = "Full rebuild" if force else "Incremental update"
            print(f"{mode}: +{added} chunks added, -{deleted} removed. Total: {total} chunks.", flush=True)

    def _status(self):
        print(self.indexer.stats(), flush=True)

    def _clear(self, sandbox_name: str):
        self.indexer.reset()
        shutil.rmtree(CHROMA_DIR, ignore_errors=True)
        shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
        clear_memory(sandbox_name)
        print("Index, output files, and conversation history cleared.", flush=True)

    def _query(self, msg: str, sandbox_name: str):
        shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

        memory = get_memory(sandbox_name)
        agent = create_agent(self.indexer, memory)

        print("Thinking...", flush=True)
        response = agent.chat(msg)
        print(str(response), flush=True)

        for path in list_output_files():
            print(f"[OUTPUT_FILE:{path}]", flush=True)

        persist_memory(sandbox_name)


# ---------------------------------------------------------------------------
# Main server loop
# ---------------------------------------------------------------------------


def main():
    print("Loading LLM...", flush=True)
    get_llm()

    print("Loading indexer...", flush=True)
    handler = MessageHandler(Indexer())

    print("Models loaded. Ready for queries.", flush=True)
    print(END_TURN, flush=True)

    handler.run()


if __name__ == "__main__":
    main()
