"""Embedding generation service."""

import logging
from typing import Iterable

import numpy as np
from sentence_transformers import SentenceTransformer

from app.core.config import get_settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Manages sentence-transformer model lifecycle and embeddings."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self._model: SentenceTransformer | None = None

    def _load_once(self) -> None:
        if self._model is not None:
            return
        logger.info("Loading embedding model: %s", self.settings.embedding_model_name)
        self._model = SentenceTransformer(
            self.settings.embedding_model_name,
            local_files_only=self.settings.local_files_only,
        )

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        """Encode texts into float32 embeddings."""
        self._load_once()
        assert self._model is not None
        embeddings = self._model.encode(
            list(texts),
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(embeddings, dtype=np.float32)
