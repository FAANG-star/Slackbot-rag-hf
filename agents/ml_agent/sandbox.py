"""ML agent sandbox — A10 GPU image with training dependencies."""

from pathlib import Path

import modal

from agents.slackbot.shared import app, data_vol, trackio_vol, TRACKIO_MOUNT
from .proxy import anthropic_proxy
from .trackio_sync import trackio_syncer

ENTRYPOINT = Path(__file__).parent / "agent.py"
CLAUDE_DIR = Path(__file__).parent / ".claude"

SANDBOX_NAME = "ml-agent"

sandbox_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "curl")
    .pip_install("claude-agent-sdk", "trackio")
    .add_local_file(ENTRYPOINT, "/agent/agent.py")
    .add_local_dir(CLAUDE_DIR, "/app/.claude")
)


def get_sandbox() -> modal.Sandbox:
    """Get the existing ML sandbox or create a new one."""
    try:
        return modal.Sandbox.from_name(app_name=app.name, name=SANDBOX_NAME)
    except modal.exception.NotFoundError:
        trackio_syncer.spawn()
        return modal.Sandbox.create(
            app=app,
            image=sandbox_image,
            workdir="/app",
            volumes={
                "/data": data_vol,
                TRACKIO_MOUNT: trackio_vol,
            },
            env={
                "ANTHROPIC_BASE_URL": anthropic_proxy.get_web_url(),
                "HF_HOME": "/data/hf-cache",
                "TRACKIO_DIR": TRACKIO_MOUNT,
            },
            secrets=[modal.Secret.from_name("github-secret")],
            gpu="A10",
            timeout=60 * 60,
            name=SANDBOX_NAME,
            idle_timeout=20 * 60,
        )
