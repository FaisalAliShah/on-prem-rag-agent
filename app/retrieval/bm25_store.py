from __future__ import annotations

import logging
import pickle
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+")
logger = logging.getLogger(__name__)


class BM25Store:
    def __init__(self, index_dir: Path):
        self.index_dir = index_dir
        self.index_path = self.index_dir / "bm25.pkl"
        self.corpus: list[str] = []
        self.metadata: list[dict[str, Any]] = []
        self.tokenized_corpus: list[list[str]] = []
        self.bm25: BM25Okapi | None = None

    def build(self, texts: list[str], metadata: list[dict[str, Any]]) -> None:
        start = time.perf_counter()
        self.corpus = texts
        self.metadata = metadata
        self.tokenized_corpus = [tokenize(text) for text in texts]
        self.bm25 = BM25Okapi(self.tokenized_corpus) if self.tokenized_corpus else None
        logger.info("bm25_built documents=%s duration_ms=%.2f", len(self.corpus), _elapsed_ms(start))

    def search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        start = time.perf_counter()
        if self.bm25 is None:
            self.load()
        if self.bm25 is None:
            logger.warning("bm25_search_empty_index")
            return []
        scores = self.bm25.get_scores(tokenize(query))
        if scores.size == 0:
            return []
        top_positions = np.argsort(scores)[::-1][:top_k]
        results: list[dict[str, Any]] = []
        for position in top_positions:
            score = float(scores[position])
            if score <= 0:
                continue
            item = dict(self.metadata[int(position)])
            item["text"] = self.corpus[int(position)]
            item["sparse_score"] = score
            item["score"] = score
            results.append(item)
        logger.info("bm25_search_completed top_k=%s results=%s duration_ms=%.2f", top_k, len(results), _elapsed_ms(start))
        return results

    def save(self) -> None:
        start = time.perf_counter()
        self.index_dir.mkdir(parents=True, exist_ok=True)
        with self.index_path.open("wb") as file:
            pickle.dump(
                {
                    "corpus": self.corpus,
                    "metadata": self.metadata,
                    "tokenized_corpus": self.tokenized_corpus,
                },
                file,
            )
        logger.info("bm25_saved path=%s documents=%s duration_ms=%.2f", self.index_path, len(self.corpus), _elapsed_ms(start))

    def load(self) -> None:
        start = time.perf_counter()
        if not self.index_path.exists():
            return
        with self.index_path.open("rb") as file:
            payload = pickle.load(file)
        self.corpus = payload["corpus"]
        self.metadata = payload["metadata"]
        self.tokenized_corpus = payload["tokenized_corpus"]
        self.bm25 = BM25Okapi(self.tokenized_corpus) if self.tokenized_corpus else None
        logger.info("bm25_loaded path=%s documents=%s duration_ms=%.2f", self.index_path, len(self.corpus), _elapsed_ms(start))


def tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_PATTERN.finditer(text)]


def _elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000
