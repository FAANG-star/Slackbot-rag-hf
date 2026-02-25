"""Slack bot — Modal app, volumes, and Bot entrypoint.

Deploy: modal deploy -m slackbot.app
"""

import os
import threading

import modal

from slackbot.modal_app import app, data_vol, rag_vol, trackio_vol, TRACKIO_MOUNT

slack_secret = modal.Secret.from_name("slack-secret")

slack_bot_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("slack-bolt", "fastapi")
)

# These imports register Modal functions/classes on `app` as a side effect.
from slackbot.index_pipeline import IndexService  # noqa: E402
from slackbot.ml_agent.service import get_sandbox  # noqa: E402
from slackbot.rag.service import RagService  # noqa: E402
from slackbot.router import Router  # noqa: E402


@app.cls(
    secrets=[slack_secret],
    image=slack_bot_image,
    volumes={"/data": rag_vol},
    min_containers=1,
    enable_memory_snapshot=True,
)
@modal.concurrent(max_inputs=20)
class Bot:
    @modal.enter(snap=True)
    def build(self):
        """Captured in the memory snapshot — only runs on first deploy."""
        from fastapi import FastAPI, Request
        from fastapi.responses import Response
        from slack_bolt import App as SlackApp
        from slack_bolt.adapter.fastapi import SlackRequestHandler

        self.router = Router(
            indexer=IndexService(),
            rag=RagService(),  # type: ignore[arg-type]
            ml_sb_fn=get_sandbox,
            vol=rag_vol,
        )

        slack_app = SlackApp(
            token=os.environ["SLACK_BOT_TOKEN"],
            signing_secret=os.environ["SLACK_SIGNING_SECRET"],
        )
        _register_slack_handlers(slack_app, self.router)

        # Slack retries events if no 200 within 3s — drop retries to avoid duplicate handling
        handler = SlackRequestHandler(slack_app)
        self._app = FastAPI()

        @self._app.post("/")
        async def root(request: Request):
            if request.headers.get("x-slack-retry-num"):
                return Response(status_code=200)
            return await handler.handle(request)

    @modal.enter(snap=False)
    def restore(self):
        """Runs on every cold start after restoring from snapshot."""
        print("Bot restored from snapshot", flush=True)

    @modal.asgi_app()
    def serve(self):
        return self._app


# ── Helpers ──────────────────────────────────────────────────────────────────


def _register_slack_handlers(slack_app, router):
    """Register Slack event handlers on the bolt app."""

    # Dispatch in a thread so Slack gets 200 within its 3s timeout
    @slack_app.event("app_mention")
    def handle_mention(body, client, **_):
        threading.Thread(
            target=router.handle,
            args=(body["event"], client),
            daemon=True,
        ).start()

    # Slack sends message events for every channel msg — ignore to avoid 404 noise
    @slack_app.event("message")
    def handle_message(**_):
        pass
