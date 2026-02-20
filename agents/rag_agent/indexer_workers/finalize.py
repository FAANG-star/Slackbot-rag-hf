"""Finalize step — merge per-worker manifests and report summary."""

from agents.slackbot.shared import app, rag_vol

from .config import index_image


@app.function(image=index_image, volumes={"/data": rag_vol}, timeout=60 * 60)
def finalize_index(force: bool = False) -> str:
    """Merge per-worker manifest files into manifest.json and report summary."""
    import json
    from pathlib import Path

    import chromadb

    CHROMA_DIR = Path("/data/rag/chroma")
    MANIFEST_PATH = CHROMA_DIR / "manifest.json"
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    if force:
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        try:
            client.delete_collection("rag_documents")
            print("Deleted existing collection.", flush=True)
        except Exception:
            pass
        for f in CHROMA_DIR.glob("manifest*.json"):
            f.unlink()
        rag_vol.commit()

    # Merge all per-worker manifests into the main manifest.
    try:
        manifest = json.loads(MANIFEST_PATH.read_text())
    except Exception:
        manifest = {}

    worker_files = sorted(CHROMA_DIR.glob("manifest-worker-*.json"))
    for wf in worker_files:
        try:
            manifest.update(json.loads(wf.read_text()))
            wf.unlink()
        except Exception:
            pass

    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    rag_vol.commit()

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        total = client.get_collection("rag_documents").count()
    except Exception:
        total = 0

    summary = f"Indexed {total:,} chunks from {len(manifest)} file(s)."
    print(summary, flush=True)
    return summary
