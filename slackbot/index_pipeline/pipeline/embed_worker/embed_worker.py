"""GPU embedding worker — TEI sidecar, returns chunks to the pipeline."""

import modal

from slackbot.app import app, rag_vol

from ... import config
from .helpers.file_parser import FileParser
from .tei_server import BATCH_SIZE, PORT, TeiServer


@app.cls(
    image=config.embed_image,
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

        self._tei = TeiServer()
        self._tei.start()
        self._splitter = TokenTextSplitter(chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
        self._parser = FileParser()
        self._http = httpx.Client(timeout=120.0)

    @modal.method()
    def embed(self, work: dict, worker_id: int) -> tuple[list, int]:
        """Parse files, chunk, embed via TEI. Returns (chunks, worker_id)."""
        docs = self._parser.parse(work)
        chunks = self._chunk_and_embed(docs)
        return chunks, worker_id

    def _chunk_and_embed(self, docs: list) -> list:
        """Split docs into chunks and embed via TEI."""
        nodes = self._splitter.get_nodes_from_documents(docs)
        if not nodes:
            return []
        texts = [n.get_content() for n in nodes]
        embeddings = self._embed_texts(texts)
        return [
            (n.node_id, emb, text, n.metadata)
            for n, emb, text in zip(nodes, embeddings, texts)
        ]

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Batch POST to TEI sidecar."""
        embeddings: list[list[float]] = []
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]
            resp = self._http.post(
                f"http://127.0.0.1:{PORT}/embed",
                json={"inputs": batch},
            )
            resp.raise_for_status()
            embeddings.extend(resp.json())
        return embeddings
