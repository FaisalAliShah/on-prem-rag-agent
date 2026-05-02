import logging
import time
from typing import Any

from app.embeddings.embedding_service import EmbeddingService
from app.retrieval.bm25_store import BM25Store
from app.retrieval.score_fusion import fuse_scores
from app.vector_store.base import BaseVectorStore


logger = logging.getLogger(__name__)


class HybridRetriever:
    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: BaseVectorStore,
        bm25_store: BM25Store,
        alpha: float,
    ):
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.bm25_store = bm25_store
        self.alpha = alpha

    def retrieve(self, query: str, top_k: int) -> list[dict[str, Any]]:
        results, _ = self.retrieve_with_debug(query, top_k)
        return results

    def retrieve_with_debug(self, query: str, top_k: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        total_start = time.perf_counter()
        logger.info("hybrid_retrieval_started query_length=%s top_k=%s alpha=%s", len(query), top_k, self.alpha)
        stage_start = time.perf_counter()
        query_embedding = self.embedding_service.embed_query(query)
        embed_ms = (time.perf_counter() - stage_start) * 1000
        stage_start = time.perf_counter()
        dense_results = self.vector_store.search(query_embedding, top_k)
        dense_ms = (time.perf_counter() - stage_start) * 1000
        stage_start = time.perf_counter()
        sparse_results = self.bm25_store.search(query, top_k)
        sparse_ms = (time.perf_counter() - stage_start) * 1000
        stage_start = time.perf_counter()
        fused = fuse_scores(dense_results, sparse_results, self.alpha, top_k)
        fusion_ms = (time.perf_counter() - stage_start) * 1000
        duration_ms = (time.perf_counter() - total_start) * 1000
        logger.info(
            "hybrid_retrieval_completed dense_results=%s sparse_results=%s fused_results=%s embed_ms=%.2f dense_ms=%.2f sparse_ms=%.2f fusion_ms=%.2f duration_ms=%.2f",
            len(dense_results),
            len(sparse_results),
            len(fused),
            embed_ms,
            dense_ms,
            sparse_ms,
            fusion_ms,
            duration_ms,
        )
        trace = {
            "alpha": self.alpha,
            "top_k": top_k,
            "counts": {
                "dense": len(dense_results),
                "sparse": len(sparse_results),
                "fused": len(fused),
            },
            "timings_ms": {
                "embedding": round(embed_ms, 2),
                "dense": round(dense_ms, 2),
                "sparse": round(sparse_ms, 2),
                "fusion": round(fusion_ms, 2),
                "total": round(duration_ms, 2),
            },
            "dense_results": [_debug_result(item) for item in dense_results[:top_k]],
            "sparse_results": [_debug_result(item) for item in sparse_results[:top_k]],
            "fused_results": [_debug_result(item) for item in fused[:top_k]],
        }
        return fused, trace


def _debug_result(item: dict[str, Any]) -> dict[str, Any]:
    text = str(item.get("text", ""))
    return {
        "chunk_id": item.get("chunk_id"),
        "source": item.get("source"),
        "score": _round_score(item.get("score")),
        "dense_score": _round_score(item.get("dense_score")),
        "sparse_score": _round_score(item.get("sparse_score")),
        "preview": text[:240],
    }


def _round_score(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return round(float(value), 6)
    except (TypeError, ValueError):
        return None
