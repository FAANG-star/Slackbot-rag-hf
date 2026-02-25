"""Modal app, volumes, and secrets — shared by all services.

This module has NO internal imports, so it can be safely imported
from anywhere without circular-dependency issues.
"""

import modal

app = modal.App("ml-agent")

data_vol = modal.Volume.from_name("sandbox-data", create_if_missing=True)
rag_vol = modal.Volume.from_name("sandbox-rag", create_if_missing=True)
trackio_vol = modal.Volume.from_name("sandbox-trackio", create_if_missing=True)

TRACKIO_MOUNT = "/root/.cache/huggingface/trackio"
