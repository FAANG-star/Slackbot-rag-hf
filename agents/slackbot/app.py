"""Slack bot entrypoint — Modal class with lifecycle-managed ASGI app."""

import modal

from agents.index_pipeline import IndexPipeline
from agents.slackbot.infra import app, rag_vol
from agents.slackbot.clients import RagClient, MlClient
from agents.slackbot.services import Documents, MessageRouter, SlackBot

import agents.ml_agent.sandbox  # noqa: F401  registers @app.function deps
import agents.rag_agent.sandbox  # noqa: F401  registers @app.function deps

slack_secret = modal.Secret.from_name("slack-secret")

slack_bot_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("slack-bolt", "fastapi")
    .add_local_python_source("agents")
    .add_local_dir("agents/ml_agent/.claude", "/root/agents/ml_agent/.claude")
    .add_local_dir("agents/rag_agent/rag", "/root/agents/rag_agent/rag")
)


@app.cls(
    secrets=[slack_secret],
    image=slack_bot_image,
    volumes={"/data": rag_vol},
    scaledown_window=10 * 60,
    min_containers=1,
)
@modal.concurrent(max_inputs=100)
class SlackBotService:
    @modal.enter()
    def setup(self):
        rag = RagClient()
        ml = MlClient()
        files = Documents(volume=rag_vol)
        indexer = IndexPipeline(volume=rag_vol)
        router = MessageRouter(rag=rag, ml=ml, files=files, indexer=indexer)
        self._app = SlackBot(router).create_fastapi_app()

    @modal.asgi_app()
    def serve(self):
        return self._app
