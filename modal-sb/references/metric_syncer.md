# Trackio Metric Syncer — Volume-Based Background Upload

The syncer runs on a separate CPU container with its own HF token. It polls a shared trackio volume for `.db` files and uploads them to an HF Space. The sandbox never needs HF credentials.

## How It Works

```
Sandbox (GPU)                  Trackio Volume                 Syncer (CPU)
─────────────                  ──────────────                 ────────────
setup_trackio() ──writes──→    space_id (tells syncer where)
trackio.init()  ──writes──→    <project>.db
trackio.log()   ──writes──→         │
                                    │  trackio_vol.reload()
                                    ├──────────reads──────────→  glob("*.db")
                                    │                            trackio.sync() ──→ HF Space
                               (every 30s)
```

## Volume Layout

```
/root/.cache/huggingface/trackio/   ← TRACKIO_MOUNT (shared volume)
  space_id                           ← written by setup_trackio(), read by syncer
  <project-name>.db                  ← trackio SQLite database
  <project-name>.lock                ← SQLite lock file
```

The syncer globs `*.db` at the volume root. The `space_id` file tells the syncer which HF Space to upload to — it's written by the agent via `setup_trackio()` mid-conversation.

## Change Detection

The syncer tracks file modification times to avoid re-syncing unchanged files:

```python
def run_syncer():
    import trackio

    mount = Path(TRACKIO_MOUNT)
    last_mtimes: dict[str, float] = {}

    while True:
        trackio_vol.reload()
        space_id = (mount / "space_id").read_text().strip() if (mount / "space_id").exists() else None
        if space_id:
            for db in mount.glob("*.db"):
                mtime = db.stat().st_mtime
                if mtime > last_mtimes.get(db.name, 0):
                    try:
                        trackio.sync(project=db.stem, space_id=space_id, force=True)
                        last_mtimes[db.name] = mtime
                        print(f"[syncer] synced '{db.stem}'", flush=True)
                    except Exception as e:
                        print(f"[syncer] error '{db.stem}': {e}", flush=True)
        time.sleep(30)
```

Key details:
- `trackio_vol.reload()` — volumes are snapshot-based; this refreshes to see new writes from the sandbox
- `space_id` is read each iteration — the syncer starts before the agent writes it, so it polls until it appears
- `db.stem` — the filename without extension becomes the project name for `trackio.sync()`
- `force=True` — overwrites the remote copy (safe since we're the only writer)
- `last_mtimes` persists across poll cycles so unchanged files are skipped
- The volume mount path matches trackio's default (`~/.cache/huggingface/trackio`) so `trackio.sync()` finds `.db` files without symlinks

**Important caveat:** `trackio.sync()` ignores the `TRACKIO_DIR` environment variable — it always looks for `.db` files in `~/.cache/huggingface/trackio/` regardless of what `TRACKIO_DIR` is set to. The `TRACKIO_DIR` env var only affects `trackio.init()` (where the `.db` is written). This is why the volume must be mounted at `TRACKIO_MOUNT = "/root/.cache/huggingface/trackio"` — so both `trackio.init()` and `trackio.sync()` agree on the path.

## Modal Function

Spawned by `create_sandbox()` before the sandbox starts, runs until the app exits:

```python
syncer_secret = modal.Secret.from_name("hf-secret")
syncer_image = modal.Image.debian_slim(python_version="3.11").pip_install("trackio")

@app.function(
    image=syncer_image,
    volumes={TRACKIO_MOUNT: trackio_vol},
    secrets=[syncer_secret],
    timeout=60 * 60,
)
def trackio_syncer():
    run_syncer()
```

- `hf-secret` provides `HF_TOKEN` for uploading to HF Spaces
- The syncer image only needs `trackio` — no ML packages
- Called via `trackio_syncer.spawn()` for background execution
- 1 hour timeout matches the sandbox timeout

## Full Implementation

```python
"""Trackio syncer — polls for .db changes and syncs metrics to HF Space."""

import time
from pathlib import Path

import modal

from .infra import app, trackio_vol, TRACKIO_MOUNT

syncer_secret = modal.Secret.from_name("hf-secret")
syncer_image = modal.Image.debian_slim(python_version="3.11").pip_install("trackio")


def run_syncer():
    """Poll for changed .db files and sync them to the HF Space."""
    import trackio

    mount = Path(TRACKIO_MOUNT)
    last_mtimes: dict[str, float] = {}

    while True:
        trackio_vol.reload()
        space_id = (mount / "space_id").read_text().strip() if (mount / "space_id").exists() else None
        if space_id:
            for db in mount.glob("*.db"):
                mtime = db.stat().st_mtime
                if mtime > last_mtimes.get(db.name, 0):
                    try:
                        trackio.sync(project=db.stem, space_id=space_id, force=True)
                        last_mtimes[db.name] = mtime
                        print(f"[syncer] synced '{db.stem}'", flush=True)
                    except Exception as e:
                        print(f"[syncer] error '{db.stem}': {e}", flush=True)
        time.sleep(30)


@app.function(
    image=syncer_image,
    volumes={TRACKIO_MOUNT: trackio_vol},
    secrets=[syncer_secret],
    timeout=60 * 60,
)
def trackio_syncer():
    run_syncer()
```
