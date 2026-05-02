import hashlib
import json
import logging
import time
from collections import Counter
from pathlib import Path

import numpy as np

from app.config import settings
from app.embeddings.embedding_service import EmbeddingService
from app.ingestion.chunker import chunk_documents
from app.ingestion.loaders import document_id_from_path, iter_document_files, load_document_files
from app.retrieval.bm25_store import BM25Store
from app.vector_store.base import BaseVectorStore


logger = logging.getLogger(__name__)
MANIFEST_NAME = "ingestion_manifest.json"
EMBEDDINGS_NAME = "embeddings.npy"


class IngestionPipeline:
    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: BaseVectorStore,
        bm25_store: BM25Store,
    ):
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.bm25_store = bm25_store

    def run(self) -> dict[str, int | float]:
        total_start = time.perf_counter()
        logger.info("ingestion_started raw_data_dir=%s", settings.raw_data_dir)
        stage_start = time.perf_counter()
        current_files = _current_file_infos(settings.raw_data_dir)
        previous_manifest = _load_manifest()
        previous_chunks = _load_previous_chunks()
        previous_embeddings = _load_previous_embeddings()
        reusable = _can_reuse_previous_state(previous_chunks, previous_embeddings, previous_manifest)
        changed_sources, skipped_sources, deleted_sources = _classify_sources(current_files, previous_manifest, reusable)
        changed_paths = [Path(info["source"]) for info in current_files.values() if info["source"] in changed_sources]
        logger.info(
            "documents_classified current_files=%s changed_files=%s skipped_files=%s deleted_files=%s reusable=%s duration_ms=%.2f",
            len(current_files),
            len(changed_sources),
            len(skipped_sources),
            len(deleted_sources),
            reusable,
            _elapsed_ms(stage_start),
        )

        stage_start = time.perf_counter()
        documents = load_document_files(changed_paths)
        logger.info("documents_loaded count=%s duration_ms=%.2f", len(documents), _elapsed_ms(stage_start))
        stage_start = time.perf_counter()
        changed_chunks = chunk_documents(documents, settings.chunk_size, settings.chunk_overlap)
        logger.info(
            "chunks_created count=%s chunk_size=%s overlap=%s duration_ms=%.2f",
            len(changed_chunks),
            settings.chunk_size,
            settings.chunk_overlap,
            _elapsed_ms(stage_start),
        )
        stage_start = time.perf_counter()
        changed_texts = [chunk["text"] for chunk in changed_chunks]
        changed_embeddings = self.embedding_service.embed_texts(changed_texts)
        logger.info("embeddings_created count=%s duration_ms=%.2f", len(changed_embeddings), _elapsed_ms(stage_start))

        stage_start = time.perf_counter()
        chunks, embeddings = _merge_chunks_and_embeddings(
            current_files=current_files,
            skipped_sources=skipped_sources,
            previous_chunks=previous_chunks,
            previous_embeddings=previous_embeddings,
            changed_chunks=changed_chunks,
            changed_embeddings=changed_embeddings,
        )
        logger.info(
            "ingestion_state_merged total_chunks=%s reused_chunks=%s changed_chunks=%s duration_ms=%.2f",
            len(chunks),
            len(chunks) - len(changed_chunks),
            len(changed_chunks),
            _elapsed_ms(stage_start),
        )

        texts = [chunk["text"] for chunk in chunks]
        metadata = [{key: value for key, value in chunk.items() if key != "text"} for chunk in chunks]

        stage_start = time.perf_counter()
        self.vector_store.add_documents(texts, embeddings, metadata)
        self.vector_store.save()
        logger.info("vector_store_saved backend=%s duration_ms=%.2f", settings.vector_backend, _elapsed_ms(stage_start))
        stage_start = time.perf_counter()
        self.bm25_store.build(texts, metadata)
        self.bm25_store.save()
        logger.info("bm25_store_saved duration_ms=%.2f", _elapsed_ms(stage_start))

        settings.metadata_dir.mkdir(parents=True, exist_ok=True)
        (settings.metadata_dir / "documents.json").write_text(
            json.dumps(_document_summaries(chunks), indent=2),
            encoding="utf-8",
        )
        _save_embeddings(embeddings)
        _save_manifest(current_files, chunks)
        _write_documents_ingested(chunks)
        duration_ms = _elapsed_ms(total_start)
        result = {
            "documents_ingested": len(current_files),
            "chunks_created": len(chunks),
            "total_documents": len(current_files),
            "total_chunks": len(chunks),
            "changed_files": len(changed_sources),
            "skipped_files": len(skipped_sources),
            "deleted_files": len(deleted_sources),
            "changed_chunks": len(changed_chunks),
            "reused_chunks": len(chunks) - len(changed_chunks),
            "duration_ms": duration_ms,
        }
        logger.info(
            "ingestion_finished documents_ingested=%s chunks_created=%s changed_files=%s skipped_files=%s deleted_files=%s duration_ms=%.2f",
            result["documents_ingested"],
            result["chunks_created"],
            len(changed_sources),
            len(skipped_sources),
            len(deleted_sources),
            duration_ms,
        )
        return result


def _document_summaries(chunks: list[dict]) -> list[dict]:
    counts = Counter(chunk["document_id"] for chunk in chunks)
    first_source: dict[str, str] = {}
    for chunk in chunks:
        first_source.setdefault(chunk["document_id"], chunk["source"])
    return [
        {"document_id": document_id, "source": first_source[document_id], "chunks": count}
        for document_id, count in sorted(counts.items())
    ]


def load_document_summaries() -> list[dict]:
    path = settings.metadata_dir / "documents.json"
    if not path.exists():
        logger.info("document_summaries_missing path=%s", path)
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def load_ingestion_manifest() -> dict:
    return _load_manifest()


def load_chunk_metadata() -> list[dict]:
    return _load_previous_chunks()


def _write_documents_ingested(chunks: list[dict]) -> None:
    summaries = _document_summaries(chunks)
    lines = ["# Documents Ingested", ""]
    if not summaries:
        lines.append("No documents ingested yet.")
    for item in summaries:
        lines.append(f"- `{item['document_id']}` from `{item['source']}`: {item['chunks']} chunks")
    Path("DOCUMENTS_INGESTED.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000


def _current_file_infos(raw_data_dir: Path) -> dict[str, dict]:
    infos: dict[str, dict] = {}
    for path in iter_document_files(raw_data_dir):
        stat = path.stat()
        source = str(path)
        infos[source] = {
            "source": source,
            "document_id": document_id_from_path(path),
            "sha256": _sha256(path),
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
    return infos


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _manifest_path() -> Path:
    return settings.metadata_dir / MANIFEST_NAME


def _embeddings_path() -> Path:
    return settings.metadata_dir / EMBEDDINGS_NAME


def _chunks_path() -> Path:
    return settings.metadata_dir / "chunks.json"


def _load_manifest() -> dict:
    path = _manifest_path()
    if not path.exists():
        return {"files": {}}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logger.exception("ingestion_manifest_load_failed path=%s", path)
        return {"files": {}}


def _load_previous_chunks() -> list[dict]:
    path = _chunks_path()
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logger.exception("previous_chunks_load_failed path=%s", path)
        return []


def _load_previous_embeddings() -> list[list[float]]:
    path = _embeddings_path()
    if not path.exists():
        return []
    try:
        embeddings = np.load(path)
    except Exception:
        logger.exception("previous_embeddings_load_failed path=%s", path)
        return []
    return np.asarray(embeddings, dtype="float32").tolist()


def _can_reuse_previous_state(
    previous_chunks: list[dict],
    previous_embeddings: list[list[float]],
    previous_manifest: dict,
) -> bool:
    return (
        bool(previous_chunks)
        and len(previous_chunks) == len(previous_embeddings)
        and previous_manifest.get("chunk_size") == settings.chunk_size
        and previous_manifest.get("chunk_overlap") == settings.chunk_overlap
        and previous_manifest.get("embedding_model") == settings.embedding_model
    )


def _classify_sources(
    current_files: dict[str, dict],
    previous_manifest: dict,
    reusable: bool,
) -> tuple[set[str], set[str], set[str]]:
    previous_files = previous_manifest.get("files", {})
    current_sources = set(current_files)
    previous_sources = set(previous_files)
    deleted_sources = previous_sources - current_sources
    skipped_sources: set[str] = set()
    changed_sources: set[str] = set()

    for source, info in current_files.items():
        previous = previous_files.get(source)
        if reusable and previous and previous.get("sha256") == info["sha256"]:
            skipped_sources.add(source)
        else:
            changed_sources.add(source)

    return changed_sources, skipped_sources, deleted_sources


def _merge_chunks_and_embeddings(
    current_files: dict[str, dict],
    skipped_sources: set[str],
    previous_chunks: list[dict],
    previous_embeddings: list[list[float]],
    changed_chunks: list[dict],
    changed_embeddings: list[list[float]],
) -> tuple[list[dict], list[list[float]]]:
    previous_by_source: dict[str, list[tuple[dict, list[float]]]] = {}
    if skipped_sources:
        for chunk, embedding in zip(previous_chunks, previous_embeddings, strict=True):
            previous_by_source.setdefault(chunk["source"], []).append((chunk, embedding))

    changed_by_source: dict[str, list[tuple[dict, list[float]]]] = {}
    for chunk, embedding in zip(changed_chunks, changed_embeddings, strict=True):
        changed_by_source.setdefault(chunk["source"], []).append((chunk, embedding))

    merged_chunks: list[dict] = []
    merged_embeddings: list[list[float]] = []
    for source in current_files:
        items = previous_by_source.get(source, []) if source in skipped_sources else changed_by_source.get(source, [])
        for chunk, embedding in items:
            merged_chunks.append(chunk)
            merged_embeddings.append(embedding)

    return merged_chunks, merged_embeddings


def _save_embeddings(embeddings: list[list[float]]) -> None:
    settings.metadata_dir.mkdir(parents=True, exist_ok=True)
    np.save(_embeddings_path(), np.asarray(embeddings, dtype="float32"))


def _save_manifest(current_files: dict[str, dict], chunks: list[dict]) -> None:
    counts = Counter(chunk["source"] for chunk in chunks)
    files = {}
    for source, info in current_files.items():
        files[source] = {**info, "chunks": counts.get(source, 0)}
    payload = {
        "version": 1,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "embedding_model": settings.embedding_model,
        "files": files,
    }
    _manifest_path().write_text(json.dumps(payload, indent=2), encoding="utf-8")
