"""Merge per-worker manifests and report index summary."""

import json
from pathlib import Path

from agents.slackbot.infra import app, rag_vol

from ..config import CHROMA_COLLECTION, CHROMA_DIR, index_image

_CHROMA_DIR = Path(CHROMA_DIR)
_MANIFEST_PATH = _CHROMA_DIR / "manifest.json"


@app.function(image=index_image, volumes={"/data": rag_vol}, timeout=60 * 60)
def finalize_index() -> str:
    import chromadb

    _CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    manifest = _merge_worker_manifests()
    chunks = chromadb.PersistentClient(path=CHROMA_DIR).get_or_create_collection(CHROMA_COLLECTION).count()

    files = f"{len(manifest)} file{'s' if len(manifest) != 1 else ''}"
    summary = f"Done! {chunks:,} searchable passages from {files}."
    print(summary, flush=True)
    return summary


def _merge_worker_manifests() -> dict:
    try:
        manifest = json.loads(_MANIFEST_PATH.read_text())
    except FileNotFoundError:
        manifest = {}

    for wf in sorted(_CHROMA_DIR.glob("manifest-worker-*.json")):
        manifest.update(json.loads(wf.read_text()))
        wf.unlink()

    _MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    rag_vol.commit()
    return manifest
