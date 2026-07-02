"""
retrieval/hybrid_search.py — BM25 sparse retrieval (complements FAISS dense search).

"""

from __future__ import annotations

import numpy as np
from rank_bm25 import BM25Okapi

class HybridSearch:
    """BM25-based sparse retrieval over the indexed text corpus."""

    def __init__(self, texts: list[str], metadata: list[dict]) -> None:
        self.texts     = texts
        self.metadata  = metadata
        self.tokenized = [t.lower().split() for t in texts]
        self.bm25      = BM25Okapi(self.tokenized)

    def bm25_search(
        self, query: str, top_k: int = 10
    ) -> tuple[list[str], list[dict]]:
        """Return (texts, metadata) for the top-k BM25 matches."""
        tokenized_query = query.lower().split()
        scores          = self.bm25.get_scores(tokenized_query)
        top_k           = min(top_k, len(self.texts))
        top_indices     = np.argsort(scores)[::-1][:top_k]
        result_texts    = [self.texts[i] for i in top_indices]
        result_meta     = [self.metadata[i] for i in top_indices]
        return result_texts, result_meta
