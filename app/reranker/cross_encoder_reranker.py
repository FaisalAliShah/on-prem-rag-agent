import logging
import time
from typing import Any

from app.config import settings


logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            logger.info(
                "reranker_model_loading model=%s local_files_only=%s cache_folder=%s",
                self.model_name,
                settings.hf_hub_offline,
                settings.hf_hub_cache,
            )
            self._model = CrossEncoder(
                self.model_name,
                cache_folder=str(settings.hf_hub_cache),
                local_files_only=settings.hf_hub_offline,
            )
            logger.info("reranker_model_loaded model=%s", self.model_name)
        return self._model

    def rerank(self, query: str, chunks: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        if not chunks:
            logger.info("rerank_skipped_empty_chunks")
            return []
        pairs = [(query, chunk["text"]) for chunk in chunks]
        try:
            start = time.perf_counter()
            scores = self.model.predict(pairs)
        except Exception:
            logger.exception("rerank_failed chunks=%s model=%s", len(chunks), self.model_name)
            raise
        reranked: list[dict[str, Any]] = []
        for chunk, score in zip(chunks, scores, strict=True):
            item = dict(chunk)
            item["rerank_score"] = float(score)
            item["score"] = float(score)
            reranked.append(item)
        results = sorted(reranked, key=lambda item: item["rerank_score"], reverse=True)[:top_k]
        logger.info(
            "rerank_completed input_chunks=%s output_chunks=%s duration_ms=%.2f",
            len(chunks),
            len(results),
            (time.perf_counter() - start) * 1000,
        )
        return results
