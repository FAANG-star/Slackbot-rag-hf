"""Volume file operations for Slack-uploaded documents."""

import os
import threading
import urllib.request
from pathlib import Path

import modal


class Documents:
    """Manages documents on the Modal volume for the RAG pipeline."""

    def __init__(self, volume: modal.Volume, docs_dir: Path = Path("/data/rag/docs")):
        self._volume = volume
        self._docs_dir = docs_dir
        self._lock = threading.Lock()

    def download_slack_files(self, files: list[dict]) -> list[str]:
        """Download Slack-shared files to volume. Returns saved filenames."""
        with self._lock:
            self._docs_dir.mkdir(parents=True, exist_ok=True)
            saved = []
            for f in files:
                url = f.get("url_private_download") or f.get("url_private")
                if not url:
                    print(f"[files] skipping file {f.get('name', '?')}: no url", flush=True)
                    continue
                filename = f.get("name", f["id"])
                print(f"[files] downloading {filename} ({f.get('size', '?')} bytes)...", flush=True)
                req = urllib.request.Request(url)
                req.add_header("Authorization", f"Bearer {os.environ['SLACK_BOT_TOKEN']}")
                with urllib.request.urlopen(req, timeout=120) as resp:
                    content = resp.read()
                print(f"[files] downloaded {len(content)} bytes", flush=True)
                dest = self._docs_dir / filename
                dest.write_bytes(content)
                saved.append(filename)
            print(f"[files] committing volume...", flush=True)
            self._volume.commit()
            print(f"[files] done: {saved}", flush=True)
            return saved

    def list_sources(self) -> str:
        """List files currently on the volume with sizes."""
        with self._lock:
            self._volume.reload()
            if not self._docs_dir.exists():
                return "No documents uploaded yet. Share files via Slack's `/drive` integration."
            files = sorted(p.name for p in self._docs_dir.iterdir() if p.is_file())
            if not files:
                return "No documents uploaded yet. Share files via Slack's `/drive` integration."
            lines = [f"Indexed sources ({len(files)}):"]
            for name in files:
                size = (self._docs_dir / name).stat().st_size
                if size > 1024 * 1024:
                    lines.append(f"  - {name} ({size / (1024 * 1024):.1f} MB)")
                else:
                    lines.append(f"  - {name} ({size / 1024:.0f} KB)")
            return "\n".join(lines)

    def remove_source(self, filename: str) -> str:
        """Remove a file from the volume."""
        with self._lock:
            self._volume.reload()
            path = self._docs_dir / filename
            if not path.exists():
                return f"File `{filename}` not found. Use `sources` to see available files."
            path.unlink()
            self._volume.commit()
            return f"Removed `{filename}`. Run `reindex` or share new files to update the search index."

    def upload_output_files(
        self, client, channel: str, thread_ts: str, file_paths: list[str]
    ):
        """Upload generated files from the sandbox volume to Slack."""
        with self._lock:
            self._volume.reload()
            for path_str in file_paths:
                path = Path(path_str)
                if path.exists() and path.stat().st_size > 0:
                    client.files_upload_v2(
                        channel=channel,
                        thread_ts=thread_ts,
                        file=str(path),
                        filename=path.name,
                        title=path.name,
                    )
