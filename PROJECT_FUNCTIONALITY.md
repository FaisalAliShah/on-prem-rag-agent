# Project Functionality

## Overview

This project is a local, on-premises Retrieval-Augmented Generation (RAG) application. It ingests local documents, indexes them for dense and sparse retrieval, reranks candidate chunks, and uses a local Ollama model to generate answers with sources.

The system is designed for local demos and small-to-medium document collections. It avoids external LLM APIs and can run fully on a single machine after required models are cached.

## Main Capabilities

### Document Ingestion

- Reads documents from `data/raw/`.
- Supports `.txt`, `.md`, and `.pdf` files.
- Extracts document text and splits it into overlapping chunks.
- Stores chunk metadata in `storage/metadata/`.
- Tracks ingested files using an ingestion manifest.
- Uses file hashes to skip unchanged documents on repeated ingestion.
- Rebuilds indexes using existing chunks and embeddings when files are unchanged.
- Removes deleted files from the rebuilt retrieval state after ingestion.

### Embeddings

- Uses Sentence Transformers for local embedding generation.
- Default embedding model: `sentence-transformers/all-MiniLM-L6-v2`.
- Stores embeddings locally for reuse during incremental ingestion.
- Supports Hugging Face cache and offline mode.

### Dense Vector Retrieval

- Supports two dense vector backends through a shared `BaseVectorStore` interface:
  - FAISS for lightweight local in-process vector search.
  - Qdrant for a vector database backend with a web dashboard.
- Backend is selected with `VECTOR_BACKEND`.
- Qdrant can be started with the Docker Compose `qdrant` profile.

### Sparse Retrieval

- Uses BM25 through `rank-bm25`.
- BM25 index is stored locally in `storage/bm25/`.
- Sparse retrieval helps with exact keyword, name, email, and URL style questions.

### Hybrid Retrieval

- Runs dense vector search and BM25 sparse search.
- Combines both result sets with score fusion.
- Uses `HYBRID_ALPHA` to control dense-vs-sparse weighting.

### Reranking

- Uses a local cross-encoder reranker.
- Default reranker model: `cross-encoder/ms-marco-MiniLM-L-6-v2`.
- Reranks fused retrieval candidates before answer generation.
- Applies a relative rerank threshold to keep only strongly relevant chunks.

### Local Answer Generation

- Uses Ollama as the local LLM provider.
- Default model: `qwen2.5:0.5b`.
- Docker Compose automatically pulls the configured Ollama model into the `ollama_data` volume.
- RAG prompt construction is handled in the app before sending context to Ollama.

### Contact Information Extraction

- Detects contact-style questions.
- Extracts emails and URLs directly from retrieved chunks when possible.
- This avoids asking the LLM to infer obvious structured contact fields.

### API Endpoints

- `GET /health`
  - Returns application health, active backend, and LLM provider.
- `POST /ingest`
  - Runs ingestion and returns changed/skipped/deleted file counts.
- `POST /query`
  - Answers a question using hybrid retrieval, reranking, and Ollama.
- `GET /documents`
  - Lists ingested documents and chunk counts.
- `GET /stats`
  - Shows backend, model, vector store, ingestion, and retrieval stats.
- `POST /warmup`
  - Loads models, indexes, vector store, BM25, reranker, and optionally Ollama.

### Query Debug Mode

`POST /query` supports:

```json
{
  "question": "your question",
  "top_k": 5,
  "debug": true
}
```

Debug mode returns:

- Dense search results.
- Sparse BM25 results.
- Fused results.
- Reranked results.
- Retrieval timing breakdown.

### Streamlit Demo UI

- Provides a basic browser UI on `http://localhost:8501`.
- Supports:
  - Chat-style querying.
  - Source display.
  - Debug trace toggle.
  - `top_k` selector.
  - Ingestion button.
  - Warm-up button.
  - Runtime stats.
  - Document upload into `data/raw/`.

### Docker Compose Runtime

- Runs the FastAPI app, Ollama, optional Qdrant, and Streamlit UI.
- Persists local runtime data through mounted folders and Docker volumes.
- Includes health checks for app, Ollama, Qdrant, and UI.
- Automatically pulls the configured Ollama model before app startup.

### Logging

- Writes application logs to:
  - `logs/app.log`
  - `logs/error.log`
- Logs request duration.
- Logs ingestion stage timings.
- Logs retrieval, reranking, vector search, BM25, and Ollama generation timings.
- Uses log rotation settings from environment variables.

## Current Architecture

```text
Documents in data/raw
        |
        v
Ingestion loader
        |
        v
Chunker
        |
        v
Embeddings + metadata
        |
        +--> Dense vector store: FAISS or Qdrant
        |
        +--> Sparse BM25 store

User query
        |
        v
Query embedding
        |
        +--> Dense search
        +--> Sparse BM25 search
        |
        v
Score fusion
        |
        v
Cross-encoder reranker
        |
        v
Prompt construction
        |
        v
Ollama answer generation
        |
        v
Answer + sources
```

## Production Improvements Not Implemented

The project is functional for local usage and demos, but several production-grade improvements were intentionally left out because of time, scope, and local resource constraints.

### Authentication and Authorization

Current state:

- API and UI are open locally.
- No user login, API keys, roles, or permissions.

Production improvement:

- Add authentication for both API and UI.
- Add role-based access for ingestion, querying, stats, and debug endpoints.
- Protect uploaded documents and logs from unauthorized access.

### Secure Secret Management

Current state:

- Configuration is loaded from `.env`.
- `.env` is excluded from Git.

Production improvement:

- Use a managed secret store such as Vault, AWS Secrets Manager, Doppler, 1Password, or Kubernetes secrets.
- Avoid putting sensitive values directly in local files on servers.

### Background Job Queue for Ingestion

Current state:

- Ingestion runs synchronously inside the API request.

Production improvement:

- Move ingestion to a background worker using Celery, RQ, Dramatiq, or a queue service.
- Add job IDs, progress tracking, retries, and failure recovery.

### Concurrent Ingestion Safety

Current state:

- The app assumes one ingestion job at a time.

Production improvement:

- Add distributed locks or database-backed ingestion state.
- Prevent two ingestion jobs from writing indexes and metadata simultaneously.

### Persistent Metadata Database

Current state:

- Metadata is stored as JSON files under `storage/metadata/`.

Production improvement:

- Move document, chunk, ingestion manifest, and audit metadata to PostgreSQL or another durable database.
- Add migrations and relational constraints.

### Object Storage for Documents

Current state:

- Documents are stored on the local filesystem under `data/raw/`.

Production improvement:

- Store original documents in S3, MinIO, Azure Blob, or another object store.
- Keep document metadata and object keys in a database.

### Stronger Document Parsing

Current state:

- PDF parsing uses a simple text extraction path.

Production improvement:

- Add OCR for scanned PDFs.
- Add table extraction.
- Preserve page numbers and section structure.
- Support more formats such as DOCX, HTML, CSV, XLSX, and emails.

### Better Chunking Strategy

Current state:

- Uses fixed-size overlapping text chunks.

Production improvement:

- Add structure-aware chunking by headings, pages, sections, and semantic boundaries.
- Tune chunk size per document type.
- Add page numbers and section labels to source citations.

### Qdrant Native Hybrid Search

Current state:

- Qdrant stores dense vectors only.
- BM25 is stored separately and fused in Python.

Production improvement:

- Optionally store dense and sparse vectors in Qdrant.
- Use Qdrant native hybrid search and filtering.
- Keep the current separate BM25 path only if backend portability is more important.

### Scalable Vector Index Lifecycle

Current state:

- Ingestion rebuilds the full selected vector index from the current chunk state.

Production improvement:

- Add incremental upserts/deletes per document.
- Add index versioning and rollback.
- Avoid full rebuilds for large corpora.

### Observability

Current state:

- Logs include useful timings and errors.

Production improvement:

- Add metrics with Prometheus/OpenTelemetry.
- Add distributed tracing.
- Add dashboards for latency, ingestion duration, model load time, error rates, and query quality.

### Centralized Logging

Current state:

- Logs are written to local files.

Production improvement:

- Ship logs to a central system such as ELK, Loki, Datadog, or CloudWatch.
- Add structured JSON logs and correlation IDs across services.

### Evaluation Harness

Current state:

- No formal regression evaluation suite is included in the cleaned runtime repo.

Production improvement:

- Add a repeatable evaluation dataset.
- Track expected source documents, answer quality, faithfulness, latency, and retrieval recall.
- Run evaluations in CI before deployments.

### Automated Tests and CI

Current state:

- Test files were removed to keep the project minimal.

Production improvement:

- Restore unit and integration tests.
- Add CI for linting, type checks, Docker build checks, and API smoke tests.
- Add test coverage for ingestion, retrieval, query responses, and error handling.

### Rate Limiting and Abuse Protection

Current state:

- No rate limits are enforced.

Production improvement:

- Add per-user/IP rate limits.
- Add request size limits for uploads.
- Add timeout and cancellation controls for expensive queries.

### Model Management

Current state:

- Model names are configured through environment variables.
- Ollama model is pulled automatically by Docker Compose.

Production improvement:

- Pin and validate model versions.
- Add model readiness checks.
- Add automated model download/cache bootstrap for embedding and reranker models.
- Add model upgrade and rollback process.

### Offline Bootstrap Package

Current state:

- A new machine needs to download Hugging Face and Ollama models unless caches already exist.

Production improvement:

- Provide an offline bundle for air-gapped environments.
- Include model artifacts, checksums, and restore scripts.

### UI Hardening

Current state:

- Streamlit UI is intended for local demos.

Production improvement:

- Add authentication.
- Add upload validation and file size limits.
- Add document delete/reingest controls.
- Add better source previews and page-level citations.
- Consider a production frontend framework if the UI becomes user-facing.

### Deployment Hardening

Current state:

- Docker Compose is used for local deployment.

Production improvement:

- Add Kubernetes manifests or Helm chart for production.
- Add resource limits.
- Add restart and readiness policies per service.
- Add separate storage classes for documents, indexes, and model caches.

### Backup and Restore

Current state:

- Runtime data lives in local folders and Docker volumes.

Production improvement:

- Add automated backups for documents, metadata, vector indexes, Qdrant data, and model caches.
- Add restore validation.

### Multi-Tenant or Multi-Collection Support

Current state:

- The app assumes one document corpus and one active backend configuration.

Production improvement:

- Add tenants, workspaces, or collections.
- Isolate documents, indexes, permissions, and query history per tenant.

## Summary

This project is a working local RAG system with ingestion, hybrid retrieval, reranking, local LLM generation, Qdrant/FAISS support, useful logging, debug mode, and a Streamlit demo UI. It is appropriate for demos, experimentation, and local knowledge-base workflows.

Before using it as a production service, the most important improvements would be authentication, background ingestion jobs, durable metadata storage, stronger document parsing, observability, automated tests, and deployment hardening.
