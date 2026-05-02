from typing import Any


def fuse_scores(
    dense_results: list[dict[str, Any]],
    sparse_results: list[dict[str, Any]],
    alpha: float,
    top_k: int,
) -> list[dict[str, Any]]:
    dense_norm = _normalize({item["chunk_id"]: float(item.get("dense_score", item["score"])) for item in dense_results})
    sparse_norm = _normalize({item["chunk_id"]: float(item.get("sparse_score", item["score"])) for item in sparse_results})

    by_chunk: dict[str, dict[str, Any]] = {}
    for item in dense_results + sparse_results:
        by_chunk.setdefault(item["chunk_id"], dict(item))

    fused: list[dict[str, Any]] = []
    for chunk_id, item in by_chunk.items():
        dense_score = dense_norm.get(chunk_id, 0.0)
        sparse_score = sparse_norm.get(chunk_id, 0.0)
        combined = alpha * dense_score + (1 - alpha) * sparse_score
        fused_item = dict(item)
        fused_item["dense_score"] = dense_score
        fused_item["sparse_score"] = sparse_score
        fused_item["score"] = combined
        fused.append(fused_item)

    return sorted(fused, key=lambda item: item["score"], reverse=True)[:top_k]


def _normalize(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    values = list(scores.values())
    minimum = min(values)
    maximum = max(values)
    if maximum == minimum:
        return {key: 1.0 for key in scores}
    return {key: (value - minimum) / (maximum - minimum) for key, value in scores.items()}

