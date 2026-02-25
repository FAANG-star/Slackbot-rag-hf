"""ML agent sandbox — image definition and sandbox factory.

Runs the agent package (Claude Agent SDK) on an A10 GPU. Support services
(API proxy, metrics syncer) are registered via the proxies package.
"""

from pathlib import Path

import modal

from slackbot.modal_app import app, data_vol, trackio_vol, TRACKIO_MOUNT
from .proxies import anthropic_proxy, trackio_syncer

SANDBOX_DIR = Path(__file__).parent / "sandbox"
CLAUDE_DIR = Path(__file__).parent / "_claude"
SANDBOX_NAME = "ml-agent"

# -- Sandbox image: Claude Agent SDK + trackio on A10 GPU --

sandbox_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "curl")
    .pip_install("claude-agent-sdk", "trackio")
    .env({"IMAGE_VERSION": "1"})
    .add_local_dir(SANDBOX_DIR, "/agent")
    .add_local_dir(CLAUDE_DIR, "/app/.claude")
)


# -- Sandbox factory --


def get_sandbox() -> modal.Sandbox:
    """Get the existing ML sandbox or create a new one."""
    try:
        return modal.Sandbox.from_name(app_name=app.name, name=SANDBOX_NAME)
    except modal.exception.NotFoundError:
        trackio_syncer.spawn()
        return modal.Sandbox.create(
            "python", "-u", "/agent/agent.py",
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
