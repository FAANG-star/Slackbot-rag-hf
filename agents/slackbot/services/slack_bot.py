"""SlackBot — sets up slack-bolt handlers, delegates to injected router."""

import os
import re
import threading
from .router import MessageContext, MessageRouter


class SlackBot:
    _SUGGESTED_PROMPTS = [
        {"title": "Show sources", "message": "sources"},
        {"title": "Reindex documents", "message": "reindex"},
        {"title": "ML training", "message": "hf: Help me fine-tune a model on HuggingFace"},
    ]

    def __init__(self, router: MessageRouter):
        self._router = router

    def create_fastapi_app(self):
        from fastapi import FastAPI, Request
        from slack_bolt import App as SlackApp, Assistant
        from slack_bolt.adapter.fastapi import SlackRequestHandler
        from starlette.responses import Response

        slack_app = SlackApp(
            token=os.environ["SLACK_BOT_TOKEN"],
            signing_secret=os.environ["SLACK_SIGNING_SECRET"],
        )
        handler = SlackRequestHandler(slack_app)
        assistant = Assistant()

        @assistant.thread_started
        def start_thread(say, set_suggested_prompts):
            say(
                "Hello! I'm an AI agent running in a Modal GPU sandbox. "
                "Share files via `/drive` to index them, then ask questions or request analysis."
            )
            set_suggested_prompts(prompts=self._SUGGESTED_PROMPTS)

        @assistant.user_message
        def handle_dm(say, set_status, set_title, payload):
            message = payload.get("text", "")
            thread_ts = payload.get("thread_ts", payload.get("ts", ""))
            set_title(title=message[:40] + ("..." if len(message) > 40 else ""))
            ctx = MessageContext(
                message=message,
                sandbox_name=f"agent-{thread_ts}".replace(".", "-"),
                thread_ts=thread_ts,
                say=say,
                set_status=lambda s: set_status(status=s),
                files=payload.get("files"),
                client=say.client if hasattr(say, "client") else None,
                channel=payload.get("channel"),
            )
            self._router.route(ctx)

        slack_app.use(assistant)

        @slack_app.event("app_mention")
        def handle_mention(body, client, **kwargs):
            event = body["event"]
            print(f"[bot] app_mention: text={event.get('text', '')[:60]!r} files={len(event.get('files', []))}", flush=True)
            message = re.sub(r"<@[A-Z0-9]+>", "", event["text"]).strip()
            channel = event["channel"]
            thread_ts = event.get("thread_ts", event["ts"])
            ctx = MessageContext(
                message=message,
                sandbox_name=f"agent-{thread_ts}".replace(".", "-"),
                thread_ts=thread_ts,
                say=lambda text: client.chat_postMessage(
                    channel=channel, text=text, thread_ts=thread_ts,
                ),
                set_status=lambda s: None,
                files=event.get("files"),
                client=client,
                channel=channel,
            )
            threading.Thread(
                target=self._router.route, args=(ctx,), daemon=True
            ).start()

        @slack_app.event("message")
        def handle_message(event, client, **kwargs):
            subtype = event.get("subtype")
            is_im = event.get("channel_type") == "im"
            print(f"[bot] message: subtype={subtype} is_im={is_im} files={len(event.get('files', []))} text={event.get('text', '')[:60]!r}", flush=True)
            # Allow: plain DMs and file_share (DM or channel)
            if subtype and subtype != "file_share":
                return
            if not is_im and not event.get("files"):
                return
            channel = event["channel"]
            thread_ts = event.get("thread_ts", event["ts"])
            message = event.get("text", "")
            ctx = MessageContext(
                message=message,
                sandbox_name=f"agent-{thread_ts}".replace(".", "-"),
                thread_ts=thread_ts,
                say=lambda text: client.chat_postMessage(
                    channel=channel, text=text, thread_ts=thread_ts,
                ),
                set_status=lambda s: None,
                files=event.get("files"),
                client=client,
                channel=channel,
            )
            threading.Thread(
                target=self._router.route, args=(ctx,), daemon=True
            ).start()

        fastapi_app = FastAPI()

        @fastapi_app.post("/")
        async def root(request: Request):
            if request.headers.get("x-slack-retry-num"):
                return Response(status_code=200)
            return await handler.handle(request)

        return fastapi_app
