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

    # ── Public API ───────────────────────────────────────────────────────────

    def is_running(self) -> bool:
        """True if a sandbox is currently connected and alive."""
        return self._sandbox is not None

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
            return self._read_response()

    # ── Sandbox lifecycle ────────────────────────────────────────────────────

    def _ensure_sandbox(self, on_status=None):
        """Attach to an existing sandbox or boot a fresh one."""
        if self._sandbox is not None:
            return
        if not self._connect_existing():
            self._boot_sandbox(on_status)

    def _connect_existing(self) -> bool:
        """Try to reconnect to a named running sandbox. Returns True on success."""
        try:
            sb = modal.Sandbox.from_name(app_name=app.name or "", name=SANDBOX_NAME)
        except modal.exception.NotFoundError:
            return False
        # Name registry lags behind termination — verify it's still alive.
        if sb.poll() is not None:
            print("[RAG] Found terminated sandbox in registry, ignoring.", flush=True)
            try:
                sb.terminate()
            except Exception:
                pass
            return False
        self._sandbox = sb
        self._stdout = iter(sb.stdout)
        print("[RAG] Reconnected to existing sandbox.", flush=True)
        return True

    def _boot_sandbox(self, on_status=None):
        """Create a new sandbox and wait for the server init sentinel."""
        if on_status:
            on_status("loading models...")
        try:
            self._sandbox = modal.Sandbox.create(
                "python", "-u", "-m", "server",
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
        except modal.exception.AlreadyExistsError:
            # Race: from_name() saw nothing, but create() found it.
            # Modal's name registry has propagation delay after termination.
            if not self._connect_existing():
                raise RuntimeError("Sandbox AlreadyExistsError but from_name() found nothing.")
            return
        self._stdout = iter(self._sandbox.stdout)
        self._wait_for_init()

    def _wait_for_init(self):
        """Stream sandbox stdout until the init sentinel."""
        assert self._stdout is not None
        try:
            for line in self._stdout:
                print(f"[RAG init] {line.rstrip()}", flush=True)
                if self._END_TURN in line:
                    return
        except Exception as exc:
            raise RuntimeError(f"RAG process crashed during init.\nstderr: {self._read_stderr()}") from exc
        raise RuntimeError(f"RAG process closed before init sentinel.\nstderr: {self._read_stderr()}")

    def _reset(self):
        """Terminate the sandbox and clear state."""
        if self._sandbox is not None:
            try:
                self._sandbox.terminate()
            except Exception:
                pass
            self._sandbox = None
        self._stdout = None

    def _read_stderr(self) -> str:
        if self._sandbox is None:
            return ""
        try:
            return self._sandbox.stderr.read()
        except Exception:
            return ""

    # ── Response parsing ─────────────────────────────────────────────────────

    def _read_response(self) -> RagResponse:
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
