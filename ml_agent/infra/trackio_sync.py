"""Trackio syncer — polls for .db changes and syncs metrics to HF Space."""

import time
from pathlib import Path

import modal

from .shared import app, trackio_vol, TRACKIO_MOUNT

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
