"""Drug name normalization using fuzzy + embeddings + FAISS."""

import logging
import time
from pathlib import Path
from typing import Dict, List

import faiss
import numpy as np
from rapidfuzz import fuzz, process

from app.core.config import get_settings
from app.services.embedding_service import EmbeddingService
from app.utils.drug_loader import load_fda_drug_names

logger = logging.getLogger(__name__)


class NormalizationService:
    """Normalizes candidate tokens against FDA drug names."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.embedding_service = EmbeddingService()
        self.drug_names: List[str] = []
        self.index: faiss.Index | None = None
        self._initialized = False

    def _load_or_build_faiss(self) -> None:
        if self._initialized:
            return

        if self.settings.local_files_only and not self.settings.drugs_cache_path.exists():
            raise FileNotFoundError(
                f"Missing local drug cache: {self.settings.drugs_cache_path}"
            )
        self.drug_names = load_fda_drug_names(
            self.settings.kaggle_dataset_ref,
            self.settings.drugs_cache_path,
            local_files_only=self.settings.local_files_only,
        )
        index_path = self.settings.faiss_index_path
        emb_path = self.settings.embeddings_cache_path
        index_path.parent.mkdir(parents=True, exist_ok=True)

        if index_path.exists() and emb_path.exists():
            logger.info("Loading FAISS index from cache")
            self.index = faiss.read_index(str(index_path))
        else:
            logger.info("Building FAISS index with %s drugs", len(self.drug_names))
            embeddings = self.embedding_service.encode(self.drug_names)
            np.save(str(emb_path), embeddings)
            self.index = faiss.IndexFlatIP(embeddings.shape[1])
            self.index.add(embeddings)
            faiss.write_index(self.index, str(index_path))

        self._initialized = True

    def _fuzzy_candidates(self, token: str, limit: int = 5) -> Dict[str, float]:
        matches = process.extract(
            token, self.drug_names, scorer=fuzz.token_sort_ratio, limit=limit
        )
        return {name: score / 100.0 for name, score, _ in matches}

    def _embedding_candidates(self, token: str, limit: int = 5) -> Dict[str, float]:
        assert self.index is not None
        vec = self.embedding_service.encode([token])
        scores, ids = self.index.search(vec, limit)
        result: Dict[str, float] = {}
        for score, idx in zip(scores[0], ids[0]):
            if 0 <= idx < len(self.drug_names):
                result[self.drug_names[idx]] = float(max(0.0, min(1.0, score)))
        return result

    def normalize(self, candidates: List[str], top_k: int | None = None) -> List[str]:
        """Return deduplicated top normalized drug names."""
        t_start = time.perf_counter()
        self._load_or_build_faiss()
        top_k = top_k or self.settings.normalize_top_k
        combined: Dict[str, float] = {}

        for token in candidates:
            fuzzy_map = self._fuzzy_candidates(token, limit=top_k)
            emb_map = self._embedding_candidates(token, limit=top_k)
            names = set(fuzzy_map) | set(emb_map)
            for name in names:
                score = 0.55 * fuzzy_map.get(name, 0.0) + 0.45 * emb_map.get(name, 0.0)
                combined[name] = max(combined.get(name, 0.0), score)

        ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        deduped = []
        seen = set()
        for name, _ in ranked:
            if name not in seen:
                seen.add(name)
                deduped.append(name)
            if len(deduped) >= top_k:
                break
        if self.settings.enable_latency_logging:
            logger.info(
                "normalize_latency_ms candidates=%d total=%.1f",
                len(candidates),
                (time.perf_counter() - t_start) * 1000,
            )
        return deduped
