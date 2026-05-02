from app.ingestion.chunker import chunk_documents
from app.ingestion.loaders import LoadedDocument


def test_chunk_documents_adds_metadata_and_stable_ids():
    document = LoadedDocument(
        document_id="policy",
        source="data/raw/policy.md",
        text=" ".join(["risk"] * 80),
        page=None,
    )

    chunks = chunk_documents([document], chunk_size=80, chunk_overlap=10)

    assert len(chunks) > 1
    assert chunks[0]["chunk_id"] == "policy_chunk_001"
    assert chunks[0]["document_id"] == "policy"
    assert chunks[0]["source"] == "data/raw/policy.md"
    assert chunks[0]["text"]


def test_chunk_overlap_must_be_smaller_than_chunk_size():
    document = LoadedDocument("doc", "doc.txt", "hello world")

    try:
        chunk_documents([document], chunk_size=10, chunk_overlap=10)
    except ValueError as exc:
        assert "chunk_overlap" in str(exc)
    else:
        raise AssertionError("Expected ValueError")

