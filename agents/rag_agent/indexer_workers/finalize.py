"""Finalize step — merge worker pkl files into ChromaDB and write manifest."""

from agents.slackbot.shared import app, rag_vol

from .config import UPSERT_BATCH, index_image


@app.function(image=index_image, volumes={"/data": rag_vol}, timeout=60 * 60)
def finalize_index(force: bool = False) -> str:
    """Merge worker pkl files into ChromaDB and write manifest."""
    import json
    import pickle
    import shutil
    from pathlib import Path

    import chromadb

    CHROMA_DIR = Path("/data/rag/chroma")
    DOCS_DIR = Path("/data/rag/docs")
    PENDING_DIR = Path("/data/rag/pending")
    MANIFEST_PATH = CHROMA_DIR / "manifest.json"
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    if force:
        try:
            client.delete_collection("rag_documents")
        except Exception:
            pass
    collection = client.get_or_create_collection("rag_documents")

    pkl_files = sorted(PENDING_DIR.glob("*.pkl")) if PENDING_DIR.exists() else []
    print(f"Merging {len(pkl_files)} worker pkl(s) into ChromaDB...", flush=True)

    total = 0
    for pkl_path in pkl_files:
        with open(pkl_path, "rb") as f:
            records = pickle.load(f)
        for i in range(0, len(records), UPSERT_BATCH):
            chunk = records[i : i + UPSERT_BATCH]
            collection.upsert(
                ids=[r["id"] for r in chunk],
                embeddings=[r["embedding"] for r in chunk],
                documents=[r["document"] for r in chunk],
                metadatas=[r["metadata"] for r in chunk],
            )
        total += len(records)
        print(f"  {pkl_path.stem}: {len(records):,} chunks (total: {total:,})", flush=True)

    manifest = {}
    if DOCS_DIR.exists():
        for path in sorted(DOCS_DIR.iterdir()):
            if path.is_file():
                stat = path.stat()
                manifest[path.name] = f"{stat.st_mtime_ns}:{stat.st_size}"
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))

    if PENDING_DIR.exists():
        shutil.rmtree(PENDING_DIR)
    rag_vol.commit()

    summary = f"Indexed {total:,} chunks from {len(manifest)} file(s)."
    print(summary, flush=True)
    return summary
