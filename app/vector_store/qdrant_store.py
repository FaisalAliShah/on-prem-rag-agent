import logging
import time
from typing import Any

from app.config import settings
from app.vector_store.base import BaseVectorStore


logger = logging.getLogger(__name__)


class QdrantVectorStore(BaseVectorStore):
    def __init__(self, url: str, collection_name: str):
        self.url = url
        self.collection_name = collection_name
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from qdrant_client import QdrantClient

            self._client = QdrantClient(url=self.url)
            logger.info("qdrant_client_initialized url=%s collection=%s", self.url, self.collection_name)
        return self._client

    def add_documents(
        self,
        texts: list[str],
        embeddings: list[list[float]],
        metadata: list[dict[str, Any]],
    ) -> None:
        start = time.perf_counter()
        if not embeddings:
            self._recreate_collection(vector_size=settings.embedding_dimension)
            logger.warning("qdrant_add_documents_empty collection=%s", self.collection_name)
            return

        vector_size = len(embeddings[0])
        self._recreate_collection(vector_size)

        from qdrant_client import models

        points = [
            models.PointStruct(
                id=index,
                vector=embedding,
                payload={**item, "text": text},
            )
            for index, (text, embedding, item) in enumerate(zip(texts, embeddings, metadata, strict=True))
        ]
        self.client.upsert(collection_name=self.collection_name, points=points, wait=True)
        logger.info(
            "qdrant_documents_added collection=%s count=%s dimension=%s duration_ms=%.2f",
            self.collection_name,
            len(points),
            vector_size,
            _elapsed_ms(start),
        )

    def search(self, query_embedding: list[float], top_k: int) -> list[dict[str, Any]]:
        start = time.perf_counter()
        if not self.client.collection_exists(self.collection_name):
            logger.warning("qdrant_search_missing_collection collection=%s", self.collection_name)
            return []

        hits = self._search(query_embedding, top_k)
        results: list[dict[str, Any]] = []
        for hit in hits:
            payload = dict(hit.payload or {})
            payload["dense_score"] = float(hit.score)
            payload["score"] = float(hit.score)
            results.append(payload)
        logger.info(
            "qdrant_search_completed collection=%s top_k=%s results=%s duration_ms=%.2f",
            self.collection_name,
            top_k,
            len(results),
            _elapsed_ms(start),
        )
        return results

    def save(self) -> None:
        logger.info("qdrant_save_noop collection=%s", self.collection_name)
        return None

    def load(self) -> None:
        exists = self.client.collection_exists(self.collection_name)
        logger.info("qdrant_load_checked collection=%s exists=%s", self.collection_name, exists)
        return None

    def _recreate_collection(self, vector_size: int) -> None:
        from qdrant_client import models

        if self.client.collection_exists(self.collection_name):
            self.client.delete_collection(collection_name=self.collection_name)
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
        )
        logger.info("qdrant_collection_recreated collection=%s dimension=%s", self.collection_name, vector_size)

    def _search(self, query_embedding: list[float], top_k: int):
        if hasattr(self.client, "query_points"):
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                limit=top_k,
                with_payload=True,
            )
            return response.points
        return self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k,
            with_payload=True,
        )


def _elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000
