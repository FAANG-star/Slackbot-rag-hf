"""Sandbox container — GPU image and creation for the Claude agent."""

from pathlib import Path

import modal

from .shared import app, data_vol, trackio_vol, TRACKIO_MOUNT
from .proxy import anthropic_proxy
from .trackio_sync import trackio_syncer

ENTRYPOINT = Path(__file__).parent.parent / "agent" / "agent.py"
CLAUDE_DIR = Path(__file__).parent.parent / ".claude"

sandbox_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "curl")
    .pip_install("claude-agent-sdk", "trackio")
    .add_local_file(ENTRYPOINT, "/agent/agent.py")
    .add_local_dir(CLAUDE_DIR, "/app/.claude")
)


def run_sandbox(image):
    """Create and return a Modal sandbox with GPU, volumes, and proxy env."""
    return modal.Sandbox.create(
        app=app,
        image=image,
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
    )


def create_sandbox():
    trackio_syncer.spawn()
    return run_sandbox(sandbox_image)
