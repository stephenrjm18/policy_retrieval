"""
semantic_cache.py — Embedding-based semantic query cache.

Repeated or semantically similar queries return instantly from cache
without re-running retrieval or LLM generation.

Cache storage: in-memory dict (LRU eviction when max size is reached).
Cache key: nearest-neighbor cosine similarity ≥ threshold to stored queries.
"""

from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from typing import Any, Optional

import numpy as np

from utils.config import CACHE_ENABLED, CACHE_MAX_ENTRIES, CACHE_SIM_THRESH
from utils.logger import get_logger

logger = get_logger(__name__)

_EMB_DIM = 384   # all-MiniLM-L6-v2 output size

# Words too generic to distinguish between policy sub-topics.
# A cache hit is rejected if the new query and cached query share NO content words.
_CACHE_STOPWORDS = frozenset([
    "what", "is", "are", "the", "a", "an", "for", "of", "to", "in",
    "and", "or", "me", "my", "i", "do", "does", "how", "which", "who",
    "where", "when", "why", "can", "could", "would", "should", "please",
    "give", "tell", "show", "find", "list", "get", "required", "need",
    "documents", "claims", "reimbursement", "treatment", "procedure",
])


def _content_words(query: str) -> frozenset:
    """Extract non-stopword tokens from a query for cache collision detection."""
    import re
    tokens = re.findall(r"[a-z]+", query.lower())
    return frozenset(t for t in tokens if t not in _CACHE_STOPWORDS)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class SemanticCache:
    """
    Thread-safe (GIL-protected for CPython) semantic cache.

    Stores (embedding, response_dict) pairs.
    On lookup: if max cosine similarity ≥ threshold -> cache hit.
    """

    def __init__(
        self,
        max_entries: int = CACHE_MAX_ENTRIES,
        sim_threshold: float = CACHE_SIM_THRESH,
        enabled: bool = CACHE_ENABLED,
    ) -> None:
        self._max     = max_entries
        self._thresh  = sim_threshold
        self._enabled = enabled
        # OrderedDict for LRU: {query_text: (embedding, response, timestamp)}
        self._store: OrderedDict[str, tuple] = OrderedDict()

    # Public API

    def get(self, query: str, query_embedding: np.ndarray) -> Optional[dict]:
        """Return cached response dict if a semantically similar query is found."""
        if not self._enabled or not self._store:
            return None

        best_key   = None
        best_score = -1.0

        for key, (emb, _response, _ts) in self._store.items():
            score = _cosine(query_embedding, emb)
            if score > best_score:
                best_score = score
                best_key   = key

        if best_score >= self._thresh and best_key is not None:
            # Secondary guard: reject if the two queries share no content words.
            # Prevents high-cosine-similarity but semantically different short queries
            # (e.g. "inpatient claims" vs "outpatient claims") from colliding.
            new_cw    = _content_words(query)
            cached_cw = _content_words(best_key)
            if new_cw and cached_cw and not (new_cw & cached_cw):
                logger.info(
                    "Cache REJECT — cosine=%.3f but no content-word overlap. "
                    "query='%s'  cached='%s'  new_cw=%s  cached_cw=%s",
                    best_score, query[:40], best_key[:40], new_cw, cached_cw,
                )
                return None

            # Move to end (most recently used)
            self._store.move_to_end(best_key)
            _emb, response, _ts = self._store[best_key]
            logger.info(
                "Cache HIT  sim=%.3f  query='%s'  matched='%s'",
                best_score,
                query[:50],
                best_key[:50],
            )
            cached = dict(response)
            cached["cached"] = True
            return cached

        return None

    def set(self, query: str, query_embedding: np.ndarray, response: dict) -> None:
        """Store a new query->response mapping."""
        if not self._enabled:
            return
        if query in self._store:
            self._store.move_to_end(query)
        else:
            if len(self._store) >= self._max:
                evicted = next(iter(self._store))
                del self._store[evicted]
                logger.debug("Cache evicted oldest entry: %s...", evicted[:40])
        self._store[query] = (query_embedding, response, time.time())
        logger.debug("Cache STORE  entries=%d  query='%s...'", len(self._store), query[:60])

    def clear(self) -> None:
        self._store.clear()
        logger.info("Semantic cache cleared.")

    @property
    def size(self) -> int:
        return len(self._store)


# Module-level singleton
_cache: Optional[SemanticCache] = None


def get_cache() -> SemanticCache:
    global _cache
    if _cache is None:
        _cache = SemanticCache()
    return _cache