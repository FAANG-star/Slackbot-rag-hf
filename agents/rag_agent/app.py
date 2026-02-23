"""RagService — Modal class with GPU memory snapshot for fast cold starts."""

import modal

from agents.slackbot.infra import app, rag_vol
from agents.rag_agent.infra import image


@app.cls(
    image=image,
    gpu="A10G",
    volumes={"/data": rag_vol},
    scaledown_window=60 * 2,
    min_containers=0,
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
    startup_timeout=600,
)
@modal.concurrent(max_inputs=1)
class RagService:
    @modal.enter(snap=True)
    def start(self):
        """Start vLLM, warm up, sleep (offload weights to CPU) — state is snapshotted."""
        import sys
        sys.path.insert(0, "/agent")
        from server.container import ServiceContainer
        self._container = ServiceContainer()
        self._container.llm.start()

    @modal.enter(snap=False)
    def wake_up(self):
        """After snapshot restore: move weights back to GPU."""
        self._container.llm.wake_up()

    @modal.method()
    def query(self, message: str, session_id: str) -> tuple[str, list[str]]:
        """Run a RAG query, return (text, output_files)."""
        return self._container.query_service.run(message, session_id)

    @modal.method()
    def command(self, cmd: str) -> str:
        """Dispatch a command (reload/reindex/status/clear), return output."""
        from server.commands import dispatch
        return dispatch(cmd, self._container)

    @modal.exit()
    def stop(self):
        self._container.llm._backend.terminate()
