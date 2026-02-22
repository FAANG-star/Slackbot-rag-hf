"""Merge per-worker manifests and report index summary."""

import json
from pathlib import Path

from agents.slackbot.infra import app, rag_vol

from ..config import CHROMA_COLLECTION, CHROMA_DIR, index_image

_CHROMA_DIR = Path(CHROMA_DIR)
_MANIFEST_PATH = _CHROMA_DIR / "manifest.json"


@app.function(image=index_image, volumes={"/data": rag_vol}, timeout=60 * 60)
def finalize_index(force: bool = False) -> str:
    import chromadb

    _CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    if force:
        _reset_collection(client)

    manifest = _merge_worker_manifests()
    chunks = client.get_or_create_collection(CHROMA_COLLECTION).count()

    files = f"{len(manifest)} file{'s' if len(manifest) != 1 else ''}"
    summary = f"Done! {chunks:,} searchable passages from {files}."
    print(summary, flush=True)
    return summary


def _reset_collection(client) -> None:
    try:
        client.delete_collection(CHROMA_COLLECTION)
        print("Deleted existing collection.", flush=True)
    except Exception:
        pass
    for f in _CHROMA_DIR.glob("manifest*.json"):
        f.unlink()
    rag_vol.commit()


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
