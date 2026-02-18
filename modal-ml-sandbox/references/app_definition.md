# Modal App — Shared Resources, Sandbox Image, and Volumes

The infrastructure is split across `shared.py` (app, volumes, constants) and `sandbox.py` (image, creation). Everything else imports from `shared.py`.

## Shared Resources (shared.py)

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

## Sandbox Image (sandbox.py)

```python
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
```

The image is built in layers:

| Layer | Purpose |
|-------|---------|
| `debian_slim(python_version="3.11")` | Base OS with Python |
| `.apt_install("git", "curl")` | System tools the agent may need |
| `.pip_install("claude-agent-sdk", "trackio")` | Claude Agent SDK + trackio for metric logging |
| `.add_local_file(ENTRYPOINT, ...)` | Agent entrypoint script |
| `.add_local_dir(CLAUDE_DIR, ...)` | Agent instructions directory (CLAUDE.md, skills, scripts) |

### Directory Layout Inside the Container

```
/agent/
  agent.py                 ← executed by sb.exec()
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
```

Key details:
- `ANTHROPIC_BASE_URL` — points the Claude SDK at the proxy instead of directly at Anthropic
- `HF_HOME` — caches models/datasets on the persistent data volume
- `TRACKIO_DIR` — tells `trackio.init()` to write `.db` files to the shared trackio volume
- `github-secret` — provides `GITHUB_TOKEN` so the agent can push code to git
- `trackio_syncer.spawn()` — starts the metric syncer in the background before creating the sandbox

## How Components Import

```python
# app.py (CLI)
from ml_agent.infra import app, create_sandbox

# infra/__init__.py
from .shared import app
from .sandbox import create_sandbox

# proxy.py, trackio_sync.py, sandbox.py
from .shared import app, data_vol, trackio_vol, TRACKIO_MOUNT
```

The `app` object is shared so all functions (proxy, syncer, sandbox) run in the same Modal app context.
