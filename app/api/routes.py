from __future__ import annotations

from functools import lru_cache
import logging
import time
from typing import Any

from fastapi import APIRouter, HTTPException

from app.api.schemas import (
    DocumentSummary,
    DocumentsResponse,
    HealthResponse,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    Source,
    StatsResponse,
    WarmupResponse,
)
from app.config import settings
from app.embeddings.embedding_service import EmbeddingService
from app.ingestion.pipeline import (
    IngestionPipeline,
    load_chunk_metadata,
    load_document_summaries,
    load_ingestion_manifest,
)
from app.llm.ollama_client import OllamaClient
from app.llm.prompts import build_rag_prompt, clean_generated_answer
from app.reranker.cross_encoder_reranker import CrossEncoderReranker
from app.retrieval.bm25_store import BM25Store
from app.retrieval.hybrid_retriever import HybridRetriever
from app.utils.contact_extraction import answer_contact_query
from app.vector_store.factory import get_vector_store


router = APIRouter()
logger = logging.getLogger(__name__)


@lru_cache
def _components():
    try:
        embedding_service = EmbeddingService(settings.embedding_model)
        vector_store = get_vector_store()
        bm25_store = BM25Store(settings.bm25_index_dir)
        reranker = CrossEncoderReranker(settings.reranker_model)
        llm = OllamaClient(
            settings.ollama_base_url,
            settings.ollama_model,
            settings.ollama_num_ctx,
            settings.ollama_num_predict,
        )
        logger.info("components_initialized backend=%s llm_model=%s", settings.vector_backend, settings.ollama_model)
        return embedding_service, vector_store, bm25_store, reranker, llm
    except Exception:
        logger.exception("components_initialization_failed")
        raise


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", backend=settings.vector_backend, llm_provider=settings.llm_provider)


@router.post("/ingest", response_model=IngestResponse)
def ingest() -> IngestResponse:
    try:
        embedding_service, vector_store, bm25_store, _, _ = _components()
        pipeline = IngestionPipeline(embedding_service, vector_store, bm25_store)
        result = pipeline.run()
        logger.info(
            "ingest_completed total_documents=%s total_chunks=%s changed_files=%s skipped_files=%s deleted_files=%s backend=%s",
            result["documents_ingested"],
            result["chunks_created"],
            result["changed_files"],
            result["skipped_files"],
            result["deleted_files"],
            settings.vector_backend,
        )
    except Exception:
        logger.exception("ingest_failed backend=%s raw_data_dir=%s", settings.vector_backend, settings.raw_data_dir)
        raise _api_error("INGESTION_FAILED", "Ingestion failed. Check logs for details.") from None
    return IngestResponse(status="success", backend=settings.vector_backend, **result)


@router.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    debug: dict[str, Any] | None = None
    try:
        embedding_service, vector_store, bm25_store, reranker, llm = _components()
        retriever = HybridRetriever(embedding_service, vector_store, bm25_store, settings.hybrid_alpha)
        if request.debug:
            candidates, retrieval_debug = retriever.retrieve_with_debug(request.question, settings.top_k_retrieval)
            debug = {"retrieval": retrieval_debug}
        else:
            candidates = retriever.retrieve(request.question, settings.top_k_retrieval)
        final_chunks = reranker.rerank(request.question, candidates, min(request.top_k, settings.top_k_reranked))
        final_chunks = _filter_relevant_chunks(final_chunks)
        if debug is not None:
            debug["reranked_results"] = [_debug_chunk(chunk) for chunk in final_chunks]
        if not final_chunks:
            return QueryResponse(
                answer="I could not find this in the ingested documents.",
                sources=[],
                backend=settings.vector_backend,
                debug=debug,
            )
        answer = answer_contact_query(request.question, final_chunks)
        if answer is None:
            answer = clean_generated_answer(llm.generate(build_rag_prompt(request.question, final_chunks)))
        logger.info(
            "query_completed question_length=%s candidates=%s final_chunks=%s backend=%s",
            len(request.question),
            len(candidates),
            len(final_chunks),
            settings.vector_backend,
        )
    except Exception:
        logger.exception("query_failed backend=%s question_length=%s", settings.vector_backend, len(request.question))
        raise _api_error("QUERY_FAILED", "Query failed. Check logs for details.") from None
    sources = [
        Source(source=chunk["source"], chunk_id=chunk["chunk_id"], score=float(chunk["score"]))
        for chunk in final_chunks
    ]
    return QueryResponse(answer=answer, sources=sources, backend=settings.vector_backend, debug=debug)


def _filter_relevant_chunks(chunks: list[dict]) -> list[dict]:
    if not chunks:
        return []
    scored_chunks = [
        (chunk, float(chunk.get("rerank_score", chunk.get("score", 0.0))))
        for chunk in chunks
    ]
    positive_scores = [score for _, score in scored_chunks if score > 0.0]
    if positive_scores:
        threshold = max(positive_scores) * settings.rerank_relative_threshold
        return [chunk for chunk, score in scored_chunks if score >= threshold]
    return chunks[:1]

@router.get("/documents", response_model=DocumentsResponse)
def documents() -> DocumentsResponse:
    try:
        summaries = [DocumentSummary(**item) for item in load_document_summaries()]
    except Exception:
        logger.exception("documents_list_failed metadata_dir=%s", settings.metadata_dir)
        raise _api_error("DOCUMENTS_LOAD_FAILED", "Could not load document summaries. Check logs for details.") from None
    return DocumentsResponse(documents=summaries)


@router.get("/stats", response_model=StatsResponse)
def stats() -> StatsResponse:
    try:
        chunks = load_chunk_metadata()
        summaries = load_document_summaries()
        manifest = load_ingestion_manifest()
        _, vector_store, bm25_store, _, _ = _components()
        return StatsResponse(
            backend=settings.vector_backend,
            llm_provider=settings.llm_provider,
            embedding_model=settings.embedding_model,
            reranker_model=settings.reranker_model,
            ollama_model=settings.ollama_model,
            total_documents=len(summaries),
            total_chunks=len(chunks),
            vector_store=_vector_store_stats(vector_store),
            ingestion=_ingestion_stats(manifest),
            retrieval={
                "top_k_retrieval": settings.top_k_retrieval,
                "top_k_reranked": settings.top_k_reranked,
                "hybrid_alpha": settings.hybrid_alpha,
                "rerank_relative_threshold": settings.rerank_relative_threshold,
                "bm25_documents": len(getattr(bm25_store, "corpus", []) or []),
            },
        )
    except Exception:
        logger.exception("stats_failed backend=%s", settings.vector_backend)
        raise _api_error("STATS_FAILED", "Stats failed. Check logs for details.") from None


@router.post("/warmup", response_model=WarmupResponse)
def warmup() -> WarmupResponse:
    try:
        result = warmup_components()
    except Exception:
        logger.exception("warmup_failed backend=%s", settings.vector_backend)
        raise _api_error("WARMUP_FAILED", "Warm-up failed. Check logs for details.") from None
    return WarmupResponse(status="success", backend=settings.vector_backend, **result)


def warmup_components() -> dict[str, Any]:
    start = time.perf_counter()
    warmed = {
        "embedding_model": False,
        "vector_store": False,
        "bm25": False,
        "reranker": False,
        "ollama": False,
    }
    embedding_service, vector_store, bm25_store, reranker, llm = _components()

    try:
        query_embedding = embedding_service.embed_query("warm up query")
        warmed["embedding_model"] = True
        vector_store.search(query_embedding, 1)
        warmed["vector_store"] = True
        bm25_store.search("warm up query", 1)
        warmed["bm25"] = True
        reranker.rerank("warm up query", [_warmup_chunk()], 1)
        warmed["reranker"] = True
        if settings.warmup_ollama:
            llm.generate("Reply with ok.")
            warmed["ollama"] = True
    except Exception:
        logger.exception("warmup_component_failed warmed=%s", warmed)
        raise

    duration_ms = (time.perf_counter() - start) * 1000
    logger.info("warmup_completed backend=%s warmed=%s duration_ms=%.2f", settings.vector_backend, warmed, duration_ms)
    return {"warmed": warmed, "duration_ms": duration_ms}


def _warmup_chunk() -> dict[str, Any]:
    return {
        "text": "warm up",
        "source": "warmup",
        "chunk_id": "warmup",
        "score": 0.0,
    }


def _vector_store_stats(vector_store: Any) -> dict[str, Any]:
    backend = settings.vector_backend.lower()
    if backend == "faiss":
        index_path = settings.faiss_index_dir / "index.faiss"
        chunks_path = settings.metadata_dir / "chunks.json"
        return {
            "backend": "faiss",
            "index_path": str(index_path),
            "index_exists": index_path.exists(),
            "metadata_path": str(chunks_path),
            "metadata_exists": chunks_path.exists(),
            "loaded_metadata": len(getattr(vector_store, "metadata", []) or []),
        }
    if backend == "qdrant":
        exists = vector_store.client.collection_exists(settings.qdrant_collection)
        point_count = 0
        vector_config = None
        if exists:
            collection = vector_store.client.get_collection(settings.qdrant_collection)
            point_count = vector_store.client.count(settings.qdrant_collection, exact=True).count
            vector_config = str(collection.config.params.vectors)
        return {
            "backend": "qdrant",
            "url": settings.qdrant_url,
            "collection": settings.qdrant_collection,
            "collection_exists": exists,
            "points": point_count,
            "vectors": vector_config,
        }
    return {"backend": backend}


def _ingestion_stats(manifest: dict[str, Any]) -> dict[str, Any]:
    files = manifest.get("files", {})
    return {
        "manifest_exists": (settings.metadata_dir / "ingestion_manifest.json").exists(),
        "manifest_version": manifest.get("version"),
        "files": len(files),
        "chunk_size": manifest.get("chunk_size"),
        "chunk_overlap": manifest.get("chunk_overlap"),
        "embedding_model": manifest.get("embedding_model"),
        "total_manifest_chunks": sum(int(item.get("chunks", 0)) for item in files.values()),
    }


def _debug_chunk(chunk: dict[str, Any]) -> dict[str, Any]:
    text = str(chunk.get("text", ""))
    return {
        "chunk_id": chunk.get("chunk_id"),
        "source": chunk.get("source"),
        "score": _round_score(chunk.get("score")),
        "rerank_score": _round_score(chunk.get("rerank_score")),
        "dense_score": _round_score(chunk.get("dense_score")),
        "sparse_score": _round_score(chunk.get("sparse_score")),
        "preview": text[:240],
    }


def _round_score(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return round(float(value), 6)
    except (TypeError, ValueError):
        return None


def _api_error(code: str, message: str, status_code: int = 500) -> HTTPException:
    return HTTPException(status_code=status_code, detail={"code": code, "message": message})
