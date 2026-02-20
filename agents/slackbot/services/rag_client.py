"""Client for the persistent RAG sandbox process."""

import json
import re
import threading
import traceback
from dataclasses import dataclass, field

import modal

from agents.rag_agent.sandbox import SANDBOX_NAME, app, rag_vol, sandbox_image


@dataclass
class RagResponse:
    text: str
    output_files: list[str] = field(default_factory=list)


class RagClient:
    """Persistent RAG sandbox lifecycle and stdin/stdout query protocol."""

    _END_TURN = "---END_TURN---"
    _OUTPUT_FILE_RE = re.compile(r"\[OUTPUT_FILE:(.+?)\]")

    def __init__(self):
        self._lock = threading.Lock()
        self._sandbox = None
        self._stdout = None

    def query(self, message: str, sandbox_name: str, on_status=None) -> RagResponse:
        """Send a query, return structured response. Retries once on failure."""
        with self._lock:
            msg = json.dumps({"message": message, "sandbox_name": sandbox_name}) + "\n"
            for attempt in range(2):
                try:
                    self._ensure_sandbox(on_status)
                    self._sandbox.stdin.write(msg.encode())
                    self._sandbox.stdin.drain()
                    break
                except Exception as e:
                    print(f"[RAG] Error (attempt {attempt + 1}): {e}", flush=True)
                    traceback.print_exc()
                    self._reset()
                    if attempt == 1:
                        raise
            return self._read_and_parse()

    def _ensure_sandbox(self, on_status=None):
        if self._sandbox is not None:
            return

        # Try to reuse an already-running sandbox — avoids 3-min cold start.
        # server.main is in a stdin read loop with no pending stdout, so
        # attaching to its stdin/stdout is safe.
        try:
            existing = modal.Sandbox.from_name(app_name=app.name, name=SANDBOX_NAME)
            self._sandbox = existing
            self._stdout = iter(existing.stdout)
            print("[RAG] Reconnected to existing sandbox", flush=True)
            return
        except modal.exception.NotFoundError:
            pass

        # No existing sandbox — boot a fresh one.
        if on_status:
            on_status("starting RAG server...")
        self._sandbox = modal.Sandbox.create(
            "python", "-u", "-m", "server.main",
            app=app,
            image=sandbox_image,
            workdir="/agent",
            volumes={"/data": rag_vol},
            gpu="A10G",
            env={"HF_HOME": "/data/hf-cache"},
            timeout=60 * 60,
            name=SANDBOX_NAME,
            idle_timeout=20 * 60,
        )
        self._stdout = iter(self._sandbox.stdout)

        received = False
        for line in self._stdout:
            text = line.rstrip()
            print(f"[RAG init] {text}", flush=True)
            if on_status and text:
                on_status(text.lower())
            if self._END_TURN in line:
                received = True
                break
        if not received:
            stderr = ""
            try:
                stderr = self._sandbox.stderr.read()
            except Exception:
                pass
            self._sandbox = None
            self._stdout = None
            raise RuntimeError(f"RAG process closed before init sentinel.\nstderr: {stderr}")

    def _reset(self):
        if self._sandbox is not None:
            try:
                self._sandbox.terminate()
            except Exception:
                pass
            self._sandbox = None
        self._stdout = None

    def _read_and_parse(self) -> RagResponse:
        """Read stdout until END_TURN, parse into RagResponse."""
        lines = []
        for line in self._stdout:
            print(f"[RAG out] {line.rstrip()}", flush=True)
            if self._END_TURN in line:
                content = line[: line.index(self._END_TURN)].strip()
                if content:
                    lines.append(content)
                break
            lines.append(line)
        print(f"[RAG] Got {len(lines)} response lines", flush=True)
        text = "\n".join(lines).strip()
        output_files = self._OUTPUT_FILE_RE.findall(text)
        display_text = self._OUTPUT_FILE_RE.sub("", text).strip()
        return RagResponse(text=display_text, output_files=output_files)
