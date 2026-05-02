# On-Prem RAG Agent Project Plan

## 1. Project Goal

Build a locally runnable **on-prem RAG agent** that supports:

- Document ingestion
- Chunking and metadata tracking
- Dense embeddings
- Sparse retrieval
- Hybrid search
- Re-ranking
- FastAPI query endpoint
- Local LLM answer generation
- Docker Compose-based local execution
- GitHub-ready repository structure
- Future switch from **FAISS** to **Qdrant** using a single configuration flag

The immediate implementation target is **FAISS**, but the architecture must remain backend-agnostic so Qdrant can be enabled later without rewriting the ingestion, retrieval, API, or LLM layers.

---

## 2. Hardware Constraint

Development machine:

```text
MacBook Air M1
```

This means the project should avoid heavy GPU-dependent models. The design should use:

- Lightweight embedding models
- Quantized local LLMs
- Small top-k retrieval values
- Cached indexes
- Docker-friendly services
- Minimal memory overhead

Recommended local LLM runtime:

```text
Ollama
```

Recommended local models:

```text
phi3:mini
mistral:7b-instruct-q4_K_M
llama3.2:3b
```

For the first working version, use a smaller model such as:

```text
phi3:mini
```

---

## 3. Functional Requirements

The system must provide the following:

### 3.1 Ingestion

The system should ingest local documents from a folder such as:

```text
data/raw/
```

Supported first-phase formats:

```text
.txt
.md
.pdf
```

Each document should be split into chunks and stored with metadata:

```json
{
  "chunk_id": "doc1_chunk_001",
  "document_id": "doc1",
  "source": "data/raw/doc1.pdf",
  "page": 1,
  "text": "chunk text here"
}
```

### 3.2 Dense Embeddings

Use a lightweight sentence-transformer model:

```text
sentence-transformers/all-MiniLM-L6-v2
```

The dense embeddings will be stored in FAISS first.

### 3.3 Sparse Retrieval

Use a sparse retriever such as:

```text
BM25
```

Recommended package:

```text
rank-bm25
```

Sparse retrieval should be saved locally so it can be reused without re-ingesting everything.

### 3.4 Hybrid Search

The retrieval layer should combine:

- Dense FAISS search
- Sparse BM25 search

Use score fusion:

```text
hybrid_score = alpha * dense_score + (1 - alpha) * sparse_score
```

Recommended starting value:

```text
alpha = 0.6
```

This gives slightly more weight to semantic retrieval while still preserving keyword precision.

### 3.5 Re-ranking

Add a re-ranking step after hybrid retrieval.

Recommended first implementation:

```text
cross-encoder/ms-marco-MiniLM-L-6-v2
```

Why this is recommended:

- Easier than full ColBERT
- Good enough for the assignment/demo
- Runs on CPU/M1 reasonably well
- Improves answer quality visibly

Pipeline:

```text
User query
   ↓
Hybrid retrieval top 20
   ↓
Cross-encoder reranking
   ↓
Final top 5 chunks
   ↓
LLM answer generation
```

### 3.6 Local LLM Answer Generation

Use Ollama as the local LLM server.

The FastAPI app should call Ollama through HTTP:

```text
http://ollama:11434/api/generate
```

For local non-Docker testing:

```text
http://localhost:11434/api/generate
```

The prompt should include:

- User question
- Retrieved context chunks
- Instruction to answer only from provided context
- Instruction to say when the answer is not available in the documents

### 3.7 FastAPI Endpoint

Minimum endpoints:

```text
GET  /health
POST /ingest
POST /query
GET  /documents
```

Expected `/query` input:

```json
{
  "question": "What is the document about?",
  "top_k": 5
}
```

Expected `/query` output:

```json
{
  "answer": "Generated answer here",
  "sources": [
    {
      "source": "data/raw/example.pdf",
      "chunk_id": "example_chunk_001",
      "score": 0.87
    }
  ],
  "backend": "faiss"
}
```

---

## 4. Architecture

```text
                     ┌────────────────────┐
                     │   Local Documents   │
                     └─────────┬──────────┘
                               │
                               ▼
                     ┌────────────────────┐
                     │ Ingestion Pipeline  │
                     │ - Load docs         │
                     │ - Chunk docs        │
                     │ - Add metadata      │
                     └─────────┬──────────┘
                               │
                               ▼
                     ┌────────────────────┐
                     │ Embedding Service   │
                     │ Dense embeddings    │
                     └─────────┬──────────┘
                               │
                               ▼
              ┌────────────────────────────────┐
              │ Vector Store Interface          │
              │ Backend selected by config flag │
              └───────────────┬────────────────┘
                              │
                 ┌────────────┴────────────┐
                 ▼                         ▼
        ┌─────────────────┐       ┌─────────────────┐
        │   FAISS Store    │       │  Qdrant Store    │
        │ First version    │       │ Future version   │
        └────────┬────────┘       └────────┬────────┘
                 │                         │
                 └────────────┬────────────┘
                              ▼
                    ┌──────────────────┐
                    │ Hybrid Retriever  │
                    │ Dense + sparse    │
                    └────────┬─────────┘
                             ▼
                    ┌──────────────────┐
                    │ Re-ranker         │
                    │ Cross-encoder     │
                    └────────┬─────────┘
                             ▼
                    ┌──────────────────┐
                    │ Local LLM         │
                    │ Ollama            │
                    └────────┬─────────┘
                             ▼
                    ┌──────────────────┐
                    │ FastAPI Response  │
                    └──────────────────┘
```

---

## 5. Backend Strategy: FAISS Now, Qdrant Later

The project should not hardcode FAISS inside the retrieval or API logic.

Instead, define a common interface:

```text
BaseVectorStore
```

Then implement:

```text
FaissVectorStore
QdrantVectorStore
```

The selected backend should come from an environment variable:

```text
VECTOR_BACKEND=faiss
```

Later, switching to Qdrant should only require:

```text
VECTOR_BACKEND=qdrant
```

### Why this design is important

FAISS is good for:

- Fast local development
- Simple demos
- Lightweight retrieval
- Low setup complexity

Qdrant is better for:

- Production-style vector database
- Metadata filtering
- Native payload storage
- Better API-based persistence
- Future multi-vector support

The project should therefore use FAISS first but be designed like a production-ready system.

---

## 6. Recommended Repository Structure

```text
on-prem-rag-agent/
│
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── config.py
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes.py
│   │   └── schemas.py
│   │
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── loaders.py
│   │   ├── chunker.py
│   │   └── pipeline.py
│   │
│   ├── embeddings/
│   │   ├── __init__.py
│   │   └── embedding_service.py
│   │
│   ├── vector_store/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── factory.py
│   │   ├── faiss_store.py
│   │   └── qdrant_store.py
│   │
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── bm25_store.py
│   │   ├── hybrid_retriever.py
│   │   └── score_fusion.py
│   │
│   ├── reranker/
│   │   ├── __init__.py
│   │   └── cross_encoder_reranker.py
│   │
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── ollama_client.py
│   │   └── prompts.py
│   │
│   └── utils/
│       ├── __init__.py
│       └── logging.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── storage/
│   ├── faiss/
│   ├── bm25/
│   └── metadata/
│
├── tests/
│   ├── test_ingestion.py
│   ├── test_retrieval.py
│   └── test_api.py
│
├── scripts/
│   ├── ingest.py
│   └── query.py
│
├── .env.example
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── README.md
└── DOCUMENTS_INGESTED.md
```

---

## 7. Configuration Design

Create `.env.example`:

```env
# Backend selection
VECTOR_BACKEND=faiss

# Retrieval settings
TOP_K_RETRIEVAL=20
TOP_K_RERANKED=5
HYBRID_ALPHA=0.6

# Embedding model
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Reranker model
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2

# Ollama settings
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://ollama:11434
OLLAMA_MODEL=phi3:mini

# Qdrant settings for future backend
QDRANT_URL=http://qdrant:6333
QDRANT_COLLECTION=rag_chunks

# Storage paths
RAW_DATA_DIR=data/raw
PROCESSED_DATA_DIR=data/processed
FAISS_INDEX_DIR=storage/faiss
BM25_INDEX_DIR=storage/bm25
METADATA_DIR=storage/metadata
```

---

## 8. Vector Store Interface

Create:

```text
app/vector_store/base.py
```

Expected design:

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any


class BaseVectorStore(ABC):
    """Common interface for all vector backends."""

    @abstractmethod
    def add_documents(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadata: List[Dict[str, Any]],
    ) -> None:
        pass

    @abstractmethod
    def search(
        self,
        query_embedding: List[float],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def save(self) -> None:
        pass

    @abstractmethod
    def load(self) -> None:
        pass
```

---

## 9. Backend Factory

Create:

```text
app/vector_store/factory.py
```

Expected logic:

```python
from app.config import settings


def get_vector_store():
    if settings.vector_backend == "faiss":
        from app.vector_store.faiss_store import FaissVectorStore
        return FaissVectorStore()

    if settings.vector_backend == "qdrant":
        from app.vector_store.qdrant_store import QdrantVectorStore
        return QdrantVectorStore()

    raise ValueError(f"Unsupported vector backend: {settings.vector_backend}")
```

---

## 10. FAISS First Implementation

FAISS should store vectors locally in:

```text
storage/faiss/
```

Metadata should be stored separately in:

```text
storage/metadata/chunks.json
```

BM25 should be stored in:

```text
storage/bm25/
```

Important FAISS design point:

FAISS stores vectors, not full documents. Therefore, the project must maintain a mapping:

```text
faiss_index_position → chunk metadata
```

Example:

```json
{
  "0": {
    "chunk_id": "doc1_chunk_001",
    "source": "data/raw/doc1.pdf",
    "text": "chunk text here"
  },
  "1": {
    "chunk_id": "doc1_chunk_002",
    "source": "data/raw/doc1.pdf",
    "text": "next chunk text here"
  }
}
```

---

## 11. Qdrant Future Implementation

Qdrant should use the same interface.

Payload should include:

```json
{
  "chunk_id": "doc1_chunk_001",
  "document_id": "doc1",
  "source": "data/raw/doc1.pdf",
  "page": 1,
  "text": "chunk text here"
}
```

The Qdrant implementation should be added later without changing:

- API routes
- Ingestion pipeline
- Reranker
- LLM client
- Prompting logic

Only the backend flag should change:

```env
VECTOR_BACKEND=qdrant
```

---

## 12. Retrieval Flow

### Step 1: User query

```text
What does the document say about risk assessment?
```

### Step 2: Embed query

Use the same dense embedding model used during ingestion.

### Step 3: Dense retrieval

Search FAISS for top-k dense matches.

### Step 4: Sparse retrieval

Search BM25 for keyword-based matches.

### Step 5: Score fusion

Combine dense and sparse results.

### Step 6: Re-ranking

Use cross-encoder reranker on query-document pairs.

### Step 7: LLM answer generation

Send final top chunks to local LLM.

### Step 8: Return answer and sources

Return:

- Answer
- Source documents
- Chunk IDs
- Retrieval scores
- Backend name

---

## 13. Dockerfile

Create:

```text
Dockerfile
```

Recommended Dockerfile:

```dockerfile
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Dockerfile maintenance notes

- Keep dependencies inside `requirements.txt`
- Avoid installing unnecessary system packages
- Use `.dockerignore` to avoid copying cache files
- Pin versions in `requirements.txt` once the system is working
- If FAISS has installation issues on ARM Docker, use one of these options:
  - Run Docker using `platform: linux/amd64`
  - Use Qdrant backend instead
  - Use a Conda/Micromamba-based image for FAISS compatibility

---

## 14. Docker Compose

Create:

```text
docker-compose.yml
```

Recommended Compose file:

```yaml
services:
  api:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: rag_api
    ports:
      - "8000:8000"
    env_file:
      - .env
    environment:
      VECTOR_BACKEND: ${VECTOR_BACKEND:-faiss}
      OLLAMA_BASE_URL: ${OLLAMA_BASE_URL:-http://ollama:11434}
      OLLAMA_MODEL: ${OLLAMA_MODEL:-phi3:mini}
      QDRANT_URL: ${QDRANT_URL:-http://qdrant:6333}
    volumes:
      - ./data:/app/data
      - ./storage:/app/storage
    depends_on:
      - ollama
    restart: unless-stopped

  ollama:
    image: ollama/ollama:latest
    container_name: rag_ollama
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    restart: unless-stopped

  qdrant:
    image: qdrant/qdrant:latest
    container_name: rag_qdrant
    profiles:
      - qdrant
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - qdrant_data:/qdrant/storage
    restart: unless-stopped

volumes:
  ollama_data:
  qdrant_data:
```

### Run with FAISS only

```bash
docker compose up --build
```

### Run with Qdrant enabled later

```bash
VECTOR_BACKEND=qdrant docker compose --profile qdrant up --build
```

---

## 15. Optional Model Pull Command

Before querying the system, pull the Ollama model:

```bash
docker exec -it rag_ollama ollama pull phi3:mini
```

Or for Mistral:

```bash
docker exec -it rag_ollama ollama pull mistral
```

For the M1 MacBook Air, start with:

```bash
docker exec -it rag_ollama ollama pull phi3:mini
```

---

## 16. Requirements File

Create:

```text
requirements.txt
```

Initial dependencies:

```txt
fastapi
uvicorn[standard]
pydantic
pydantic-settings
python-dotenv
sentence-transformers
rank-bm25
numpy
scikit-learn
requests
pypdf
faiss-cpu
qdrant-client
```

If `faiss-cpu` causes installation issues on Apple Silicon Docker, keep the code architecture unchanged and use one of these alternatives:

```text
1. Build/run with linux/amd64 emulation
2. Use a Micromamba Docker image and install FAISS through conda-forge
3. Temporarily switch VECTOR_BACKEND=qdrant
```

---

## 17. .dockerignore

Create:

```text
.dockerignore
```

Recommended content:

```text
.git
__pycache__
*.pyc
*.pyo
*.pyd
.env
.venv
venv
storage/faiss/*
storage/bm25/*
storage/metadata/*
.DS_Store
.ipynb_checkpoints
```

Note:

Do not ignore the full `storage/` directory if you want indexes to persist locally during development. Only ignore generated files from the Docker build context.

---

## 18. API Design

### GET `/health`

Returns service status.

Example response:

```json
{
  "status": "ok",
  "backend": "faiss",
  "llm_provider": "ollama"
}
```

### POST `/ingest`

Triggers ingestion from `data/raw/`.

Example response:

```json
{
  "status": "success",
  "documents_ingested": 5,
  "chunks_created": 132,
  "backend": "faiss"
}
```

### POST `/query`

Example request:

```json
{
  "question": "What are the main requirements in the document?",
  "top_k": 5
}
```

Example response:

```json
{
  "answer": "The document explains...",
  "sources": [
    {
      "source": "data/raw/example.pdf",
      "chunk_id": "example_chunk_003",
      "score": 0.91
    }
  ],
  "backend": "faiss"
}
```

### GET `/documents`

Returns the list of documents ingested.

Example response:

```json
{
  "documents": [
    {
      "document_id": "example_pdf",
      "source": "data/raw/example.pdf",
      "chunks": 24
    }
  ]
}
```

---

## 19. Prompt Template

Use a strict RAG prompt:

```text
You are an on-prem document question-answering assistant.
Answer the user's question using only the provided context.
If the answer is not present in the context, say: "I could not find this in the ingested documents."
Do not invent facts.

Question:
{question}

Context:
{context}

Answer:
```

This is important because it makes the system safer and easier to evaluate.

---

## 20. Implementation Milestones

### Milestone 1: Project skeleton

Deliverables:

- Folder structure
- `.env.example`
- `requirements.txt`
- Dockerfile
- Docker Compose
- FastAPI health endpoint

### Milestone 2: Document ingestion

Deliverables:

- Load `.txt`, `.md`, `.pdf`
- Chunk documents
- Save metadata
- Create `DOCUMENTS_INGESTED.md`

### Milestone 3: Dense FAISS retrieval

Deliverables:

- Generate embeddings
- Store embeddings in FAISS
- Save and reload FAISS index
- Return top-k dense chunks

### Milestone 4: Sparse BM25 retrieval

Deliverables:

- Build BM25 index
- Save/load BM25 corpus
- Return top-k sparse chunks

### Milestone 5: Hybrid search

Deliverables:

- Combine dense and sparse results
- Normalize scores
- Apply weighted fusion
- Return merged top-k chunks

### Milestone 6: Re-ranking

Deliverables:

- Add cross-encoder reranker
- Compare retrieval before/after reranking
- Return rerank scores in debug mode

### Milestone 7: Local LLM answer generation

Deliverables:

- Ollama client
- Prompt template
- Context formatting
- Generated answer with sources

### Milestone 8: API completion

Deliverables:

- `/ingest`
- `/query`
- `/documents`
- `/health`

### Milestone 9: Dockerized local run

Deliverables:

- `docker compose up --build`
- Ollama running locally
- Persistent storage volumes
- README commands

### Milestone 10: Qdrant-ready design

Deliverables:

- `QdrantVectorStore` placeholder or partial implementation
- Backend factory already working
- `VECTOR_BACKEND` flag documented
- Qdrant service available in Docker Compose profile

---

## 21. Evaluation Plan

For the demo, prepare 5 to 10 test questions.

For each question, record:

```text
Question
Expected source document
Top dense result
Top sparse result
Top hybrid result
Top reranked result
Final answer quality
```

Create a simple evaluation file:

```text
evaluation/results.md
```

Suggested metrics:

```text
Relevant @1
Relevant in Top 3
Top1 Hit Rate
Top3 Hit Rate
Answer groundedness
```

This will make the project look stronger because it proves the retrieval improvement rather than only showing the final answer.

---

## 22. Required Submission Files

The final GitHub repository should include:

```text
README.md
Dockerfile
docker-compose.yml
requirements.txt
.env.example
DOCUMENTS_INGESTED.md
app/
data/raw/ sample documents or instructions
storage/ instructions, not necessarily committed indexes
evaluation/results.md
```

Do not commit large model files or generated indexes unless specifically required.

Instead, document how to regenerate them:

```bash
docker compose up --build
curl -X POST http://localhost:8000/ingest
```

---

## 23. README Run Commands

The README should include these commands.

### 1. Copy environment file

```bash
cp .env.example .env
```

### 2. Start services

```bash
docker compose up --build
```

### 3. Pull local LLM model

```bash
docker exec -it rag_ollama ollama pull phi3:mini
```

### 4. Add documents

Place documents inside:

```text
data/raw/
```

### 5. Run ingestion

```bash
curl -X POST http://localhost:8000/ingest
```

### 6. Ask a question

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the key points in the documents?", "top_k": 5}'
```

### 7. Run with Qdrant later

```bash
VECTOR_BACKEND=qdrant docker compose --profile qdrant up --build
```

---

## 24. Interview Explanation

Use this explanation:

```text
I designed the RAG system with a backend-agnostic vector store interface. The first implementation uses FAISS because it is lightweight and suitable for local development on an M1 MacBook. However, the system is structured so that Qdrant can be enabled later using a single VECTOR_BACKEND flag. This keeps the ingestion, retrieval, reranking, API, and LLM layers independent from the vector database choice.

The system performs hybrid retrieval by combining dense semantic search with sparse BM25 search. After retrieving candidate chunks, it applies a cross-encoder reranker to improve precision before passing the final context to a locally served quantized LLM through Ollama. The whole system is runnable through Docker Compose and includes persistent local storage for indexes and metadata.
```

---

## 25. Final Delivery Checklist

Before submission, verify:

```text
[ ] GitHub repository is uploaded
[ ] README has clear setup commands
[ ] Dockerfile builds successfully
[ ] docker-compose.yml starts API and Ollama
[ ] /health endpoint works
[ ] /ingest endpoint works
[ ] /query endpoint returns answer and sources
[ ] Dense FAISS search works
[ ] Sparse BM25 search works
[ ] Hybrid fusion works
[ ] Reranking works
[ ] Local LLM generation works
[ ] DOCUMENTS_INGESTED.md is included
[ ] Sample documents or ingestion instructions are included
[ ] VECTOR_BACKEND flag is documented
[ ] Qdrant service is included in Docker Compose profile
[ ] Evaluation examples are included
```

---

## 26. Recommended Build Order

Do not try to build everything at once.

Follow this order:

```text
1. FastAPI health endpoint
2. Document loader
3. Chunking
4. Dense embeddings
5. FAISS save/load
6. BM25 save/load
7. Hybrid retrieval
8. Reranker
9. Ollama answer generation
10. Docker Compose run
11. README and final polish
12. Qdrant-ready placeholder
```

This gives a working system early and reduces debugging risk.

---

## 27. Final Recommended Strategy

For the current deadline:

```text
Build FAISS fully.
Keep Qdrant available through the architecture and Docker profile.
Document the Qdrant switch clearly.
Do not spend too much time implementing advanced Qdrant multi-vector search unless the FAISS version is already complete.
```

This gives the best balance of:

- Working demo
- Clean architecture
- Local M1 performance
- Interview-level design
- Future production readiness
