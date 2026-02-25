"""Handle file uploads — download to volume and trigger reindex."""

import os
import urllib.request
from pathlib import Path

_DOCS_DIR = Path("/data/rag/docs")


class IndexHandler:
    """Download Slack-uploaded files to the rag volume and trigger reindexing."""

    def __init__(self, indexer, vol):
        self._indexer = indexer
        self._vol = vol

    def handle(self, files: list[dict], channel: str, thread_ts: str, say) -> None:
        saved = self._download(files)
        self._vol.commit()
        if not saved:
            say("No downloadable files found in the shared items.")
            return
        say(f"Saved {len(saved)} file(s): {', '.join(saved)}")
        self._reindex(channel, thread_ts, say)

    def _download(self, files: list[dict]) -> list[str]:
        _DOCS_DIR.mkdir(parents=True, exist_ok=True)
        saved = []
        for f in files:
            url = f.get("url_private_download") or f.get("url_private")
            if not url:
                continue
            filename = f.get("name", f["id"])
            req = urllib.request.Request(url)
            req.add_header("Authorization", f"Bearer {os.environ['SLACK_BOT_TOKEN']}")
            with urllib.request.urlopen(req, timeout=120) as resp:
                (_DOCS_DIR / filename).write_bytes(resp.read())
            saved.append(filename)
        return saved

    def _reindex(self, channel: str, thread_ts: str, say) -> None:
        result = self._indexer.reindex(
            force=False, slack={"channel": channel, "thread_ts": thread_ts},
        )
        say(result or "Indexing started — I'll post updates here as it progresses.")
