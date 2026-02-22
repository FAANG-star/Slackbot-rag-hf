"""Modal sandbox lifecycle — connect, boot, reset."""

import modal

from agents.rag_agent.sandbox import SANDBOX_NAME, app, rag_vol, sandbox_image

from .config import END_TURN


class SandboxManager:
    """Manages a persistent Modal sandbox with auto-reconnect."""

    def __init__(self):
        self.sandbox: modal.Sandbox | None = None
        self.stdout = None

    @property
    def is_running(self) -> bool:
        return self.sandbox is not None

    def ensure_running(self, on_status=None) -> None:
        """Attach to existing sandbox or boot a new one."""
        if self.sandbox is not None:
            return
        if not self._try_connect():
            self._boot(on_status)

    def reset(self) -> None:
        """Terminate sandbox and clear state."""
        if self.sandbox is not None:
            try:
                self.sandbox.terminate()
            except Exception:
                pass
        self.sandbox = None
        self.stdout = None

    def _try_connect(self) -> bool:
        """Reconnect to a named running sandbox."""
        try:
            sb = modal.Sandbox.from_name(app_name=app.name or "", name=SANDBOX_NAME)
        except modal.exception.NotFoundError:
            return False

        # Name registry can lag behind termination — verify it's alive
        if sb.poll() is not None:
            print("[RAG] Found terminated sandbox, ignoring.", flush=True)
            try:
                sb.terminate()
            except Exception:
                pass
            return False

        self.sandbox = sb
        self.stdout = iter(sb.stdout)
        print("[RAG] Reconnected to existing sandbox.", flush=True)
        return True

    def _boot(self, on_status=None) -> None:
        """Create a new sandbox and wait for init sentinel."""
        if on_status:
            on_status("loading models...")
        try:
            self.sandbox = modal.Sandbox.create(
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
            # Race: from_name saw nothing but create found it
            if not self._try_connect():
                raise RuntimeError("AlreadyExistsError but from_name found nothing.")
            return

        self.stdout = iter(self.sandbox.stdout)
        self._wait_for_init()

    def _wait_for_init(self) -> None:
        """Stream stdout until the init sentinel appears."""
        for line in self.stdout:
            print(f"[RAG init] {line.rstrip()}", flush=True)
            if END_TURN in line:
                return
        stderr = ""
        if self.sandbox:
            try:
                stderr = self.sandbox.stderr.read()
            except Exception:
                pass
        raise RuntimeError(f"RAG process closed before init.\nstderr: {stderr}")
