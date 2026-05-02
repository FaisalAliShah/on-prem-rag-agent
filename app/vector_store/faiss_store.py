import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from app.vector_store.base import BaseVectorStore


logger = logging.getLogger(__name__)


class FaissVectorStore(BaseVectorStore):
    def __init__(self, index_dir: Path, metadata_dir: Path):
        self.index_dir = index_dir
        self.metadata_dir = metadata_dir
        self.index_path = self.index_dir / "index.faiss"
        self.mapping_path = self.metadata_dir / "chunks.json"
        self.index = None
        self.metadata: list[dict[str, Any]] = []

    def add_documents(
        self,
        texts: list[str],
        embeddings: list[list[float]],
        metadata: list[dict[str, Any]],
    ) -> None:
        if not embeddings:
            self.index = None
            self.metadata = []
            logger.warning("faiss_add_documents_empty")
            return
        faiss = _import_faiss()
        vectors = np.asarray(embeddings, dtype="float32")
        faiss.normalize_L2(vectors)
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)
        self.metadata = [{**item, "text": text} for item, text in zip(metadata, texts, strict=True)]
        logger.info("faiss_documents_added count=%s dimension=%s", len(self.metadata), vectors.shape[1])

    def search(self, query_embedding: list[float], top_k: int) -> list[dict[str, Any]]:
        start = time.perf_counter()
        if self.index is None:
            self.load()
        if self.index is None or not self.metadata:
            logger.warning("faiss_search_empty_index")
            return []
        faiss = _import_faiss()
        query = np.asarray([query_embedding], dtype="float32")
        faiss.normalize_L2(query)
        limit = min(top_k, len(self.metadata))
        scores, positions = self.index.search(query, limit)
        results: list[dict[str, Any]] = []
        for score, position in zip(scores[0], positions[0], strict=True):
            if position < 0:
                continue
            item = dict(self.metadata[int(position)])
            item["dense_score"] = float(score)
            item["score"] = float(score)
            results.append(item)
        logger.info("faiss_search_completed top_k=%s results=%s duration_ms=%.2f", top_k, len(results), _elapsed_ms(start))
        return results

    def save(self) -> None:
        start = time.perf_counter()
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        if self.index is not None:
            faiss = _import_faiss()
            faiss.write_index(self.index, str(self.index_path))
        self.mapping_path.write_text(json.dumps(self.metadata, indent=2), encoding="utf-8")
        logger.info(
            "faiss_saved index_path=%s metadata_path=%s chunks=%s duration_ms=%.2f",
            self.index_path,
            self.mapping_path,
            len(self.metadata),
            _elapsed_ms(start),
        )

    def load(self) -> None:
        start = time.perf_counter()
        if self.mapping_path.exists():
            self.metadata = json.loads(self.mapping_path.read_text(encoding="utf-8"))
        if self.index_path.exists():
            faiss = _import_faiss()
            self.index = faiss.read_index(str(self.index_path))
        logger.info(
            "faiss_loaded index_exists=%s metadata_count=%s duration_ms=%.2f",
            self.index is not None,
            len(self.metadata),
            _elapsed_ms(start),
        )


def _import_faiss():
    try:
        import faiss
    except ImportError as exc:
        raise RuntimeError("FAISS backend requires faiss-cpu to be installed.") from exc
    return faiss


def _elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000
