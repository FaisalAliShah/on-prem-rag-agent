# On-Prem RAG Agent

Local RAG service with document ingestion, FAISS dense retrieval, BM25 sparse retrieval, hybrid fusion, cross-encoder reranking, and Ollama answer generation.

## Stack

- FastAPI for HTTP endpoints
- Pydantic Settings for environment configuration
- Sentence Transformers for embeddings and reranking
- FAISS for local dense vector search
- rank-bm25 for sparse retrieval
- Ollama for local LLM generation
- Docker Compose for local services

## Run

```bash
cp .env.example .env
docker compose up --build
```

Docker Compose starts Ollama and automatically pulls `OLLAMA_MODEL` into the `ollama_data` volume on first run.

Place `.txt`, `.md`, or `.pdf` files in:

```text
data/raw/
```

Run ingestion:

```bash
curl -X POST http://localhost:8000/ingest
```

Ask a question:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the key points in the documents?", "top_k": 5}'
```

List ingested documents:

```bash
curl http://localhost:8000/documents
```

Health check:

```bash
curl http://localhost:8000/health
```

Open the demo UI:

```bash
docker compose up -d --build ui
```

Then visit:

```text
http://localhost:8501
```

Runtime stats:

```bash
curl http://localhost:8000/stats
```

Warm up models and indexes before a demo:

```bash
curl -X POST http://localhost:8000/warmup
```

Inspect retrieval details for a query:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the key points in the documents?", "top_k": 5, "debug": true}'
```

## Qdrant Switch

The code is structured around a `BaseVectorStore` interface. FAISS and Qdrant are both available as dense vector backends. BM25, hybrid fusion, reranking, and Ollama generation are shared by both backends.

```bash
VECTOR_BACKEND=qdrant docker compose --profile qdrant up --build
```

Run ingestion after switching backends so the selected vector store is populated:

```bash
curl -X POST http://localhost:8000/ingest
```

## Local Development

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
uvicorn app.main:app --reload
```

PyTorch is installed from the official CPU wheel index before the rest of the dependencies so local and Docker installs do not pull CUDA/NVIDIA packages.

For local non-Docker Ollama:

```env
OLLAMA_BASE_URL=http://localhost:11434
```

## Troubleshooting

If `/query` returns an Ollama memory error, use a smaller model in `.env`:

```env
OLLAMA_MODEL=qwen2.5:0.5b
OLLAMA_NUM_CTX=1024
OLLAMA_NUM_PREDICT=256
```

Then restart the API. Docker Compose will pull the configured model if it is missing:

```bash
docker compose up -d --build app
```

`RERANK_RELATIVE_THRESHOLD=0.8` keeps every reranked chunk whose score is at least 80% of the best positive rerank score. This is a relative threshold because cross-encoder rerank scores are not percentages.

## Tests

```bash
pytest
```

The tests cover pure application logic and avoid model downloads.

## Logs

Application logs are written to:

```text
logs/app.log
logs/error.log
```

The files rotate using `LOG_MAX_BYTES` and `LOG_BACKUP_COUNT` from `.env`. In Docker, `./logs` is mounted into the API container so logs persist after restarts.

`WARMUP_ON_STARTUP=1` loads the embedding model, vector backend, BM25, reranker, and Ollama during API startup. Set `WARMUP_OLLAMA=0` if you want startup warm-up to skip the LLM generation call.

## Offline Model Cache

The API uses the Docker volume `hf_cache` for Sentence Transformers models and runs with Hugging Face offline mode enabled by default:

```env
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
HF_HUB_DISABLE_TELEMETRY=1
```

If you change embedding or reranker models, temporarily set `HF_HUB_OFFLINE=0` and `TRANSFORMERS_OFFLINE=0`, run one ingestion/query to populate the cache, then switch them back to `1`.
