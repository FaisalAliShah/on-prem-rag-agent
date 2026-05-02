from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path


SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LoadedDocument:
    document_id: str
    source: str
    text: str
    page: int | None = None


def load_documents(raw_data_dir: Path) -> list[LoadedDocument]:
    return load_document_files(iter_document_files(raw_data_dir))


def iter_document_files(raw_data_dir: Path) -> list[Path]:
    if not raw_data_dir.exists():
        logger.warning("raw_data_dir_missing path=%s", raw_data_dir)
        return []
    return [
        path
        for path in sorted(raw_data_dir.rglob("*"))
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]


def load_document_files(paths: list[Path]) -> list[LoadedDocument]:
    documents: list[LoadedDocument] = []
    for path in sorted(paths):
        try:
            loaded = _load_file(path)
        except Exception:
            logger.exception("document_load_failed path=%s", path)
            continue
        logger.info("document_loaded path=%s pages_or_files=%s", path, len(loaded))
        documents.extend(loaded)
    return documents


def _load_file(path: Path) -> list[LoadedDocument]:
    document_id = document_id_from_path(path)
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return [
            LoadedDocument(
                document_id=document_id,
                source=str(path),
                text=path.read_text(encoding="utf-8", errors="ignore"),
            )
        ]
    if suffix == ".pdf":
        return _load_pdf(path, document_id)
    return []


def _load_pdf(path: Path, document_id: str) -> list[LoadedDocument]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("PDF ingestion requires the pypdf package.") from exc

    try:
        reader = PdfReader(str(path))
    except Exception:
        logger.exception("pdf_reader_failed path=%s", path)
        raise
    documents: list[LoadedDocument] = []
    for index, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            logger.exception("pdf_page_extract_failed path=%s page=%s", path, index)
            continue
        if text.strip():
            documents.append(
                LoadedDocument(
                    document_id=document_id,
                    source=str(path),
                    text=text,
                    page=index,
                )
            )
    return documents


def document_id_from_path(path: Path) -> str:
    return path.stem.lower().replace(" ", "_").replace("-", "_")
