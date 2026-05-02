from app.ingestion.loaders import LoadedDocument


def chunk_documents(
    documents: list[LoadedDocument],
    chunk_size: int,
    chunk_overlap: int,
) -> list[dict]:
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks: list[dict] = []
    counters: dict[str, int] = {}
    for document in documents:
        text = _normalize_text(document.text)
        if not text:
            continue
        start = 0
        step = chunk_size - chunk_overlap
        while start < len(text):
            raw_chunk = text[start : start + chunk_size]
            chunk_text = _trim_to_word_boundary(raw_chunk, is_last=start + chunk_size >= len(text))
            if chunk_text:
                counters[document.document_id] = counters.get(document.document_id, 0) + 1
                chunk_number = counters[document.document_id]
                chunks.append(
                    {
                        "chunk_id": f"{document.document_id}_chunk_{chunk_number:03d}",
                        "document_id": document.document_id,
                        "source": document.source,
                        "page": document.page,
                        "text": chunk_text,
                    }
                )
            start += step
    return chunks


def _normalize_text(text: str) -> str:
    return " ".join(text.split())


def _trim_to_word_boundary(text: str, is_last: bool) -> str:
    text = text.strip()
    if is_last or len(text) < 80:
        return text
    last_space = text.rfind(" ")
    if last_space <= 0:
        return text
    return text[:last_space].strip()

