# Modal App — Shared Resources, Sandbox Image, and Volumes

The infrastructure is split across `modal_app.py` (app, volumes, constants) and `service.py` (image, creation). Everything else imports from `modal_app.py`.

## Shared Resources (modal_app.py)

```python
"""Shared Modal app and volumes."""

import modal

app = modal.App("ml-agent")

data_vol = modal.Volume.from_name("sandbox-data", create_if_missing=True)
trackio_vol = modal.Volume.from_name("sandbox-trackio", create_if_missing=True)

TRACKIO_MOUNT = "/root/.cache/huggingface/trackio"
```

- `app` — single Modal app shared by proxy, syncer, and sandbox
- `data_vol` — persistent storage for HF model/dataset cache and training outputs
- `trackio_vol` — dedicated volume for trackio `.db` files and `space_id` config
- `TRACKIO_MOUNT` — matches trackio's default path so `trackio.sync()` finds `.db` files without symlinks

## Sandbox Image (service.py)

```python
from pathlib import Path
import modal
from my_package.modal_app import app, data_vol, trackio_vol, TRACKIO_MOUNT
from .proxies import anthropic_proxy, trackio_syncer

SANDBOX_DIR = Path(__file__).parent / "sandbox"
CLAUDE_DIR = Path(__file__).parent / "_claude"
SANDBOX_NAME = "ml-agent"

sandbox_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "curl")
    .pip_install("claude-agent-sdk", "trackio")
    .env({"IMAGE_VERSION": "1"})
    .add_local_dir(SANDBOX_DIR, "/agent")
    .add_local_dir(CLAUDE_DIR, "/app/.claude")
)
```

The image is built in layers:

| Layer | Purpose |
|-------|---------|
| `debian_slim(python_version="3.12")` | Base OS with Python |
| `.apt_install("git", "curl")` | System tools the agent may need |
| `.pip_install("claude-agent-sdk", "trackio")` | Claude Agent SDK + trackio for metric logging |
| `.env({"IMAGE_VERSION": "1"})` | Cache buster — bump to force code layer rebuild (must come BEFORE `add_local_dir`) |
| `.add_local_dir(SANDBOX_DIR, ...)` | Agent entrypoint directory |
| `.add_local_dir(CLAUDE_DIR, ...)` | Agent instructions directory (CLAUDE.md, skills, scripts) |

### Directory Layout Inside the Container

```
/agent/
  agent.py                 ← executed by Sandbox.create()
/app/                      ← workdir
  .claude/
    CLAUDE.md              ← agent reads this as project instructions
    scripts/setup_trackio.py ← agent calls this at runtime
    skills/                ← agent skills (transformers, trackio)
/data/
  runs/                    ← training outputs (metrics, checkpoints)
  hf-cache/                ← HuggingFace model/dataset cache
/root/.cache/huggingface/trackio/
  *.db                     ← trackio metric databases (shared volume)
  space_id                 ← tells the syncer where to upload
```

## Sandbox Creation

```python
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
```

Key details:
- `from_name` — reuses an existing sandbox if one is already running (avoids duplicate GPU allocation)
- `"python", "-u", "/agent/agent.py"` — the sandbox entrypoint command (`-u` disables output buffering)
- `ANTHROPIC_BASE_URL` — points the Claude SDK at the proxy instead of directly at Anthropic
- `HF_HOME` — caches models/datasets on the persistent data volume
- `TRACKIO_DIR` — tells `trackio.init()` to write `.db` files to the shared trackio volume
- `github-secret` — provides `GITHUB_TOKEN` so the agent can push code to git
- `trackio_syncer.spawn()` — starts the metric syncer in the background before creating the sandbox
- `name=SANDBOX_NAME` — names the sandbox so `from_name` can find it later
- `idle_timeout=20 * 60` — auto-terminates after 20 minutes of no stdin activity

## How Components Import

```python
# proxies/anthropic.py, proxies/trackio.py, service.py
from my_package.modal_app import app, data_vol, trackio_vol, TRACKIO_MOUNT
```

The `app` object is shared so all functions (proxy, syncer, sandbox) run in the same Modal app context.
