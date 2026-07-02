"""
retrieval/reranker.py — Cross-encoder reranking.

Model is lazy-loaded once per process.
Falls back to identity ranking if sentence_transformers is unavailable.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np

from utils.config import RERANKER_ENABLED, RERANKER_HUB_NAME, RERANKER_LOCAL_PATH
from utils.logger import get_logger

logger = get_logger(__name__)

_model = None
_available: Optional[bool] = None   # None = not yet checked

def _get_model():
    global _model, _available
    if _available is False:
        return None
    if _model is not None:
        return _model
    if not RERANKER_ENABLED:
        _available = False
        logger.info("Reranker disabled via config.")
        return None
    try:
        from sentence_transformers import CrossEncoder

        model_path = (
            RERANKER_LOCAL_PATH
            if os.path.isdir(RERANKER_LOCAL_PATH)
            else RERANKER_HUB_NAME
        )
        logger.info("Loading cross-encoder from: %s", model_path)
        _model = CrossEncoder(model_path)
        _available = True
        logger.info("Cross-encoder loaded.")
        return _model
    except Exception as exc:
        logger.warning("Reranker unavailable (%s) — using score-passthrough.", exc)
        _available = False
        return None

def rerank(
    query: str,
    docs: list[str],
    metadata: list[dict],
    top_k: int = 5,
) -> list[tuple[str, dict, float]]:
    """
    Rerank (doc, meta) pairs using a cross-encoder.

    Returns list of (doc, meta, score) tuples, sorted best-first,
    truncated to top_k.

    Falls back to original order with uniform score=1.0 if the
    cross-encoder model is unavailable.
    """
    if not docs:
        return []

    model = _get_model()
    if model is None:
        # No reranker — return original order, uniform scores
        return [(d, m, 1.0) for d, m in zip(docs, metadata)][:top_k]

    pairs  = [[query, doc] for doc in docs]
    scores = model.predict(pairs)
    # Sigmoid normalisation -> [0, 1]
    scores = 1.0 / (1.0 + np.exp(-scores))

    ranked = sorted(
        zip(docs, metadata, scores),
        key=lambda x: x[2],
        reverse=True,
    )
    return list(ranked)[:top_k]
