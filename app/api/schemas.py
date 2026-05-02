from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    backend: str
    llm_provider: str


class IngestResponse(BaseModel):
    status: str
    documents_ingested: int
    chunks_created: int
    backend: str
    total_documents: int
    total_chunks: int
    changed_files: int
    skipped_files: int
    deleted_files: int
    changed_chunks: int
    reused_chunks: int
    duration_ms: float


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    debug: bool = False


class Source(BaseModel):
    source: str
    chunk_id: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]
    backend: str
    debug: dict[str, Any] | None = None


class DocumentSummary(BaseModel):
    document_id: str
    source: str
    chunks: int


class DocumentsResponse(BaseModel):
    documents: list[DocumentSummary]


class StatsResponse(BaseModel):
    backend: str
    llm_provider: str
    embedding_model: str
    reranker_model: str
    ollama_model: str
    total_documents: int
    total_chunks: int
    vector_store: dict[str, Any]
    ingestion: dict[str, Any]
    retrieval: dict[str, Any]


class WarmupResponse(BaseModel):
    status: str
    backend: str
    warmed: dict[str, bool]
    duration_ms: float
