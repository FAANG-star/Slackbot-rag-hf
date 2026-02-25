"""GPU embedding worker — TEI sidecar, returns chunks to the pipeline."""

import modal

from slackbot.modal_app import app, rag_vol

from .helpers.file_parser import FileParser
from .tei_server import BATCH_SIZE, PORT, TeiServer

WORKERS_PER_GPU = 4
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 128

# TEI base image + parsing/chunking libs
embed_image = (
    modal.Image.from_registry(
        "ghcr.io/huggingface/text-embeddings-inference:86-1.7",
        add_python="3.12",
    )
    .entrypoint([])
    .uv_pip_install(
        "llama-index-core",
        "llama-index-readers-file",
        "pypdf",
        "python-docx",
        "httpx",
    )
    .run_commands("python -c \"import nltk; nltk.download('punkt_tab'); nltk.download('stopwords')\"")
)

hf_secret = modal.Secret.from_name("hf-secret")


@app.cls(
    image=embed_image,
    volumes={"/data": rag_vol},
    secrets=[hf_secret],
    gpu="A10G",
    timeout=60 * 60,
    env={
        "HF_HOME": "/data/hf-cache",
        "HUGGINGFACE_HUB_CACHE": "/data/hf-cache",
        "WORKER_VERSION": "24",
    },
)
@modal.concurrent(max_inputs=WORKERS_PER_GPU)
class EmbedWorker:

    @modal.enter()
    def _setup(self):
        import httpx
        from llama_index.core.node_parser import TokenTextSplitter

        self._tei = TeiServer()
        self._tei.start()
        self._splitter = TokenTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
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
