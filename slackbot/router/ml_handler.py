"""Handle hf: messages — run prompt in the ML sandbox."""

import json
import threading

END_SENTINEL = "__END__"


class MlHandler:
    """Run prompts in the ML training sandbox via stdin/stdout."""

    def __init__(self, sb_fn):
        self._get_sb = sb_fn
        self._sandbox = None
        self._stdout = None
        self._lock = threading.Lock()

    def handle(self, prompt: str, thread_ts: str, say) -> None:
        """Send a prompt to the GPU sandbox and relay the response back to Slack."""
        session = f"agent-{thread_ts}".replace(".", "-")
        request = json.dumps({"message": prompt, "session": session})

        with self._lock:
            self._ensure_sandbox()
            self._sandbox.stdin.write(request + "\n")
            self._sandbox.stdin.drain()

            lines = []
            for line in self._stdout:
                if END_SENTINEL in line:
                    break
                lines.append(line)

        response = "\n".join(lines).strip()
        say(response or "(No response from agent)")

    def _is_alive(self):
        # poll() returns None while running, exit code when done
        if self._sandbox is None:
            return False
        if self._sandbox.poll() is not None:
            return False
        return True

    def _ensure_sandbox(self):
        if not self._is_alive():
            self._sandbox = self._get_sb()
            # Reuse a single iterator across turns to maintain position
            self._stdout = iter(self._sandbox.stdout)
