"""Parallel indexing workers — GPU embedding + CPU finalize."""

from .config import N_WORKERS
from .embed_worker import EmbedWorker
from .finalize import finalize_index
