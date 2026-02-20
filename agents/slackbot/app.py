"""Slack bot entrypoint — Modal class with lifecycle-managed ASGI app."""

import modal

from agents.slackbot.shared import app, rag_vol
import agents.ml_agent.sandbox  # noqa: F401  registers @app.function deps
import agents.rag_agent.sandbox  # noqa: F401  registers @app.function deps
import agents.rag_agent.indexer_workers  # noqa: F401  registers GPU worker functions

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
        from agents.slackbot.bot import SlackBot
        from agents.slackbot.services.container import ServiceContainer

        self._app = SlackBot(ServiceContainer()).create_fastapi_app()

    @modal.asgi_app()
    def serve(self):
        return self._app
