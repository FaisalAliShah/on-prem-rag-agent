from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    vector_backend: str = Field(default="faiss", alias="VECTOR_BACKEND")

    top_k_retrieval: int = Field(default=20, alias="TOP_K_RETRIEVAL")
    top_k_reranked: int = Field(default=5, alias="TOP_K_RERANKED")
    rerank_relative_threshold: float = Field(default=0.8, alias="RERANK_RELATIVE_THRESHOLD")
    hybrid_alpha: float = Field(default=0.6, alias="HYBRID_ALPHA")
    warmup_on_startup: bool = Field(default=True, alias="WARMUP_ON_STARTUP")
    warmup_ollama: bool = Field(default=True, alias="WARMUP_OLLAMA")

    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        alias="EMBEDDING_MODEL",
    )
    embedding_dimension: int = Field(default=384, alias="EMBEDDING_DIMENSION")
    reranker_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        alias="RERANKER_MODEL",
    )
    hf_home: Path = Field(default=Path("/root/.cache/huggingface"), alias="HF_HOME")
    hf_hub_cache: Path = Field(default=Path("/root/.cache/huggingface/hub"), alias="HF_HUB_CACHE")
    hf_hub_offline: bool = Field(default=True, alias="HF_HUB_OFFLINE")
    hf_hub_disable_telemetry: bool = Field(default=True, alias="HF_HUB_DISABLE_TELEMETRY")

    llm_provider: str = Field(default="ollama", alias="LLM_PROVIDER")
    ollama_base_url: str = Field(default="http://ollama:11434", alias="OLLAMA_BASE_URL")
    ollama_model: str = Field(default="qwen2.5:0.5b", alias="OLLAMA_MODEL")
    ollama_num_ctx: int = Field(default=1024, alias="OLLAMA_NUM_CTX")
    ollama_num_predict: int = Field(default=256, alias="OLLAMA_NUM_PREDICT")

    qdrant_url: str = Field(default="http://qdrant:6333", alias="QDRANT_URL")
    qdrant_collection: str = Field(default="rag_chunks", alias="QDRANT_COLLECTION")

    raw_data_dir: Path = Field(default=Path("data/raw"), alias="RAW_DATA_DIR")
    faiss_index_dir: Path = Field(default=Path("storage/faiss"), alias="FAISS_INDEX_DIR")
    bm25_index_dir: Path = Field(default=Path("storage/bm25"), alias="BM25_INDEX_DIR")
    metadata_dir: Path = Field(default=Path("storage/metadata"), alias="METADATA_DIR")

    log_dir: Path = Field(default=Path("logs"), alias="LOG_DIR")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_max_bytes: int = Field(default=5_242_880, alias="LOG_MAX_BYTES")
    log_backup_count: int = Field(default=5, alias="LOG_BACKUP_COUNT")

    chunk_size: int = Field(default=900, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=150, alias="CHUNK_OVERLAP")


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
