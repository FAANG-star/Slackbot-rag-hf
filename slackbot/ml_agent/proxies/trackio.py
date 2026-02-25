"""Trackio syncer — polls for .db changes and syncs metrics to HF Space."""

import time
from pathlib import Path

import modal

from slackbot.modal_app import app, trackio_vol, TRACKIO_MOUNT

syncer_image = modal.Image.debian_slim(python_version="3.12").pip_install("trackio")
syncer_secret = modal.Secret.from_name("hf-secret")


@app.function(
    image=syncer_image,
    volumes={TRACKIO_MOUNT: trackio_vol},
    secrets=[syncer_secret],
    timeout=60 * 60,
)
def trackio_syncer():
    import trackio

    mount = Path(TRACKIO_MOUNT)
    last_mtimes: dict[str, float] = {}

    # Poll loop: check for new/changed .db files every 30s
    while True:
        trackio_vol.reload()

        # space_id is written by the sandbox when trackio.init() is called
        space_id = (mount / "space_id").read_text().strip() if (mount / "space_id").exists() else None
        if space_id:
            for db in mount.glob("*.db"):
                mtime = db.stat().st_mtime
                if mtime > last_mtimes.get(db.name, 0):
                    # Sync changed db to the HF Space dashboard
                    try:
                        trackio.sync(project=db.stem, space_id=space_id, force=True)
                        last_mtimes[db.name] = mtime
                        print(f"[syncer] synced '{db.stem}'", flush=True)
                    except Exception as e:
                        print(f"[syncer] error '{db.stem}': {e}", flush=True)
        time.sleep(30)
