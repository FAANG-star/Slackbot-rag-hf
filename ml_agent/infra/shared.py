"""Shared Modal app and volumes."""

import modal

app = modal.App("ml-agent")

data_vol = modal.Volume.from_name("sandbox-data", create_if_missing=True)
trackio_vol = modal.Volume.from_name("sandbox-trackio", create_if_missing=True)

TRACKIO_MOUNT = "/root/.cache/huggingface/trackio"
