"""Posts indexing progress to Slack — single updating message + final post."""

import os

_API = "https://slack.com/api"


class SlackNotifier:
    """No-ops silently when slack_info is None or token is missing."""

    def __init__(self, slack_info: dict | None):
        self._slack = slack_info
        self._token = os.environ.get("SLACK_BOT_TOKEN", "")
        self._ts: str | None = None

    def update(self, text: str) -> None:
        """Post or update a single progress message."""
        if self._ts:
            self._call("chat.update", {"ts": self._ts, "text": text})
        else:
            data = self._call("chat.postMessage", {"text": text})
            if data and data.get("ok"):
                self._ts = data["ts"]

    def post(self, text: str) -> None:
        """Post a new message (for final result)."""
        self._call("chat.postMessage", {"text": text})

    def _call(self, method: str, extra: dict) -> dict | None:
        if not self._slack or not self._token:
            return None
        try:
            import httpx

            resp = httpx.post(
                f"{_API}/{method}",
                headers={"Authorization": f"Bearer {self._token}", "Content-Type": "application/json"},
                json={"channel": self._slack["channel"], "thread_ts": self._slack.get("thread_ts"), **extra},
            )
            data = resp.json()
            if not data.get("ok"):
                print(f"upsert-worker: slack {method} error: {data.get('error')}", flush=True)
            return data
        except Exception as e:
            print(f"upsert-worker: slack {method} failed: {e}", flush=True)
            return None
