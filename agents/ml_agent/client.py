"""Client for the one-shot ML sandbox agent."""

from agents.ml_agent import get_sandbox


class MlClient:
    """Executes single-turn ML agent interactions."""

    def run(self, message: str, sandbox_name: str) -> str:
        """Run one ML agent turn, return the full response text."""
        sb = get_sandbox()
        process = sb.exec(
            "python",
            "-u",
            "/agent/agent.py",
            "--message",
            message,
            "--sandbox-name",
            sandbox_name,
        )

        lines = list(process.stdout)
        exit_code = process.wait()
        print(f"ML agent process exited with status {exit_code}")

        stderr = process.stderr.read()
        if stderr:
            lines.append(f"*** ERROR ***\n{stderr}")

        return "\n".join(lines).strip()
