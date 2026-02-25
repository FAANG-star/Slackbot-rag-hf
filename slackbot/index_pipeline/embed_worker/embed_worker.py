"""GPU embedding worker — TEI sidecar, sends chunks to UpsertWorker."""

import modal

from slackbot.app import app, rag_vol

from .. import config
from .helpers.file_parser import FileParser
from .helpers.tei_server import TeiServer


@app.cls(
    image=config.index_image,
    volumes={"/data": rag_vol},
    secrets=[config.hf_secret],
    gpu="A10G",
    timeout=60 * 60,
    env={
        "HF_HOME": "/data/hf-cache",
        "HUGGINGFACE_HUB_CACHE": "/data/hf-cache",
        "WORKER_VERSION": "23",
    },
)
@modal.concurrent(max_inputs=config.WORKERS_PER_GPU)
class EmbedWorker:

    @modal.enter()
    def _setup(self):
        import httpx
        from llama_index.core.node_parser import TokenTextSplitter

        self._tei = TeiServer(config.EMBEDDING_MODEL, config.TEI_PORT, config.TEI_MAX_BATCH)
        self._tei.start()
        self._splitter = TokenTextSplitter(chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
        self._parser = FileParser(embed_fn=self._chunk_and_embed)
        self._http = httpx.Client(timeout=120.0)

    @modal.method()
    def embed(self, work: dict, worker_id: int) -> tuple[int, int]:
        """Parse, chunk, embed, fire chunks to upsert worker. Returns (worker_id, chunk_count)."""
        from slackbot.index_pipeline.upsert_worker import UpsertWorker
        total, chunks = self._parser.embed(work)
        UpsertWorker().receive.spawn(chunks, worker_id, work)
        return worker_id, total

    def _chunk_and_embed(self, docs: list) -> tuple[int, list]:
        """Split docs into chunks, embed via TEI, return (count, chunk_tuples)."""
        nodes = self._splitter.get_nodes_from_documents(docs)
        if not nodes:
            return 0, []
        texts = [n.get_content() for n in nodes]
        embeddings = self._embed_texts(texts)
        chunks = [
            (n.node_id, emb, text, n.metadata)
            for n, emb, text in zip(nodes, embeddings, texts)
        ]
        return len(nodes), chunks

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Batch POST to TEI sidecar, return embeddings."""
        embeddings: list[list[float]] = []
        for i in range(0, len(texts), config.TEI_BATCH_SIZE):
            batch = texts[i : i + config.TEI_BATCH_SIZE]
            resp = self._http.post(
                f"http://127.0.0.1:{config.TEI_PORT}/embed",
                json={"inputs": batch},
            )
            resp.raise_for_status()
            embeddings.extend(resp.json())
        return embeddings
