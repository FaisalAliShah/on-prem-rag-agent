from app.retrieval.score_fusion import fuse_scores


def test_fuse_scores_combines_dense_and_sparse_results():
    dense = [
        {"chunk_id": "a", "text": "alpha", "source": "a.md", "dense_score": 0.8, "score": 0.8},
        {"chunk_id": "b", "text": "beta", "source": "b.md", "dense_score": 0.2, "score": 0.2},
    ]
    sparse = [
        {"chunk_id": "b", "text": "beta", "source": "b.md", "sparse_score": 10.0, "score": 10.0},
        {"chunk_id": "c", "text": "gamma", "source": "c.md", "sparse_score": 1.0, "score": 1.0},
    ]

    fused = fuse_scores(dense, sparse, alpha=0.4, top_k=3)

    assert {item["chunk_id"] for item in fused} == {"a", "b", "c"}
    assert fused[0]["chunk_id"] == "b"
