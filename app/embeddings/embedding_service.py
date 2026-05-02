import logging
import time

import numpy as np

from app.config import settings


logger = logging.getLogger(__name__)


class EmbeddingService:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info(
                "embedding_model_loading model=%s local_files_only=%s cache_folder=%s",
                self.model_name,
                settings.hf_hub_offline,
                settings.hf_hub_cache,
            )
            self._model = SentenceTransformer(
                self.model_name,
                cache_folder=str(settings.hf_hub_cache),
                local_files_only=settings.hf_hub_offline,
            )
            logger.info("embedding_model_loaded model=%s", self.model_name)
        return self._model

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            logger.warning("embed_texts_empty")
            return []
        try:
            start = time.perf_counter()
            embeddings = self.model.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        except Exception:
            logger.exception("embed_texts_failed count=%s model=%s", len(texts), self.model_name)
            raise
        logger.info(
            "embed_texts_completed count=%s model=%s duration_ms=%.2f",
            len(texts),
            self.model_name,
            (time.perf_counter() - start) * 1000,
        )
        return np.asarray(embeddings, dtype="float32").tolist()

    def embed_query(self, query: str) -> list[float]:
        return self.embed_texts([query])[0]
