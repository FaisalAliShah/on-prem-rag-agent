from app.config import settings
from app.vector_store.base import BaseVectorStore


def get_vector_store() -> BaseVectorStore:
    backend = settings.vector_backend.lower()
    if backend == "faiss":
        from app.vector_store.faiss_store import FaissVectorStore

        return FaissVectorStore(settings.faiss_index_dir, settings.metadata_dir)
    if backend == "qdrant":
        from app.vector_store.qdrant_store import QdrantVectorStore

        return QdrantVectorStore(settings.qdrant_url, settings.qdrant_collection)
    raise ValueError(f"Unsupported vector backend: {settings.vector_backend}")
