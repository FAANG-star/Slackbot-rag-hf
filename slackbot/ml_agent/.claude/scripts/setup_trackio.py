"""Trackio setup — inits tracking and writes space_id for the syncer.

Metrics are written to the default trackio path (~/.cache/huggingface/trackio)
which is a shared volume. A syncer function outside the sandbox reads the
space_id file and handles uploading to the HF Space.
"""

import time
from pathlib import Path

TRACKIO_DIR = Path.home() / ".cache/huggingface/trackio"


def setup_trackio(project: str, space_id: str) -> str:
    """Init trackio and write space_id for the syncer. Returns the run directory path."""
    import trackio

    run_name = f"{project}-{int(time.time())}"
    run_dir = Path("/data/runs") / run_name
    (run_dir / "checkpoint").mkdir(parents=True, exist_ok=True)

    # Write space_id so the syncer knows where to push
    (TRACKIO_DIR / "space_id").write_text(space_id)

    trackio.init(project=project, name=run_name)
    print(f"[trackio] Run dir: {run_dir}", flush=True)
    print(f"[trackio] Syncing to: {space_id}", flush=True)
    return str(run_dir)
