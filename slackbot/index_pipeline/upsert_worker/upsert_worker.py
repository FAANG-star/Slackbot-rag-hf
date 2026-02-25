"""CPU upsert worker — receives chunks from GPU workers and writes to ChromaDB.

Flow: begin(n, slack) → n × receive() via .spawn() → drain thread auto-finalizes.
"""

import queue
import threading

import modal

from slackbot.app import app, rag_vol
from slackbot.index_pipeline import config

from .helpers.chunk_store import ChunkStore
from .helpers.manifest import Manifest
from .helpers.slack_notifier import SlackNotifier


@app.cls(
    image=config.index_image,
    volumes={"/data": rag_vol},
    secrets=[modal.Secret.from_name("slack-secret")],
    scaledown_window=30 * 60,
    timeout=60 * 60,
)
@modal.concurrent(max_inputs=64)
class UpsertWorker:

    @modal.enter()
    def _setup(self):
        self._store = ChunkStore(config.CHROMA_DIR, config.CHROMA_COLLECTION)
        self._manifest = Manifest(config.CHROMA_DIR, rag_vol)
        self._notifier = SlackNotifier(None)
        self._queue: queue.Queue = queue.Queue()
        self._expected = 0
        self._received = 0
        self._recv_lock = threading.Lock()
        self._recv_reported: set[int] = set()
        self._drained = 0
        self._drain_reported: set[int] = set()
        self._drain_thread = threading.Thread(target=self._drain, daemon=True)
        self._drain_thread.start()

    @modal.method()
    def begin(self, n_batches: int, slack: dict | None = None) -> None:
        """Tell the worker how many receive() calls to expect."""
        print(f"upsert-worker: begin(n={n_batches}, slack={bool(slack)})", flush=True)
        self._expected = n_batches
        self._received = 0
        self._recv_reported = set()
        self._drained = 0
        self._drain_reported = set()
        self._notifier = SlackNotifier(slack)

    @modal.method()
    def receive(self, chunks: list, worker_id: int, work: dict) -> None:
        """Enqueue chunks for serial upsert and return immediately."""
        self._queue.put((chunks, worker_id, work))
        with self._recv_lock:
            self._received += 1
            self._report_recv()

    @modal.method()
    def reset(self) -> None:
        """Drop and recreate the collection (for force reindex)."""
        self._store.reset()
        self._manifest.clear()
        rag_vol.commit()

    # ── Internal ────────────────────────────────────────────────────────

    def _drain(self) -> None:
        """Serial upsert loop — auto-finalizes after expected batch count."""
        while True:
            try:
                chunks, worker_id, work = self._queue.get()
                self._store.upsert(chunks, worker_id)
                self._manifest.write_worker(worker_id, work)
                self._drained += 1
                self._report_drain()
                if self._expected and self._drained >= self._expected:
                    self._finalize()
            except Exception as e:
                print(f"upsert-worker: drain error: {e}", flush=True)

    def _report_recv(self) -> None:
        """Report embed completion progress (called from receive, under lock)."""
        pct = int(self._received / self._expected * 100) if self._expected else 0
        print(f"upsert-worker: embedded {self._received}/{self._expected} ({pct}%)", flush=True)
        if not self._expected:
            return
        for threshold in (25, 50, 75):
            if pct >= threshold and threshold not in self._recv_reported:
                self._recv_reported.add(threshold)
                self._notifier.update(f"Embedding ({pct}%)...")

    def _report_drain(self) -> None:
        """Report database write progress (called from drain thread)."""
        pct = int(self._drained / self._expected * 100) if self._expected else 0
        print(f"upsert-worker: written {self._drained}/{self._expected} ({pct}%)", flush=True)
        if not self._expected:
            return
        for threshold in (25, 50, 75):
            if pct >= threshold and threshold not in self._drain_reported:
                self._drain_reported.add(threshold)
                self._notifier.update(f"Writing to database ({pct}%)...")

    def _finalize(self) -> None:
        print("upsert-worker: finalizing index...", flush=True)
        manifest = self._manifest.merge()
        count = self._store.count()
        files = f"{len(manifest)} file{'s' if len(manifest) != 1 else ''}"
        result = f"Done! {count:,} searchable passages from {files}."
        print(f"upsert-worker: {result}", flush=True)

        self._notifier.post(result)
