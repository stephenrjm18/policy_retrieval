"""
services/embedding_service.py — Module-level embedding model singleton.

Loaded once at startup, reused for all queries.
Prevents per-request model loading overhead.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np

from utils.config import MODELS_DIR
from utils.logger import get_logger

logger = get_logger(__name__)

_model = None
_LOCAL_PATH = str(MODELS_DIR / "all-MiniLM-L6-v2")


def _get_model():
    global _model
    if _model is not None:
        return _model

    try:
        from sentence_transformers import SentenceTransformer

        model_path = _LOCAL_PATH if os.path.isdir(_LOCAL_PATH) else "all-MiniLM-L6-v2"
        logger.info("Loading embedding model from: %s", model_path)
        _model = SentenceTransformer(model_path)
        logger.info("Embedding model loaded.")
    except Exception as exc:
        logger.error("Failed to load embedding model: %s", exc)
        raise RuntimeError(f"Embedding model unavailable: {exc}") from exc

    return _model


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed a list of texts; returns float32 ndarray of shape (N, dim)."""
    return _get_model().encode(texts, show_progress_bar=False)


def embed_query(query: str) -> np.ndarray:
    """Embed a single query string; returns 1D float32 ndarray."""
    return embed_texts([query])[0]


def is_available() -> bool:
    """Return True if the embedding model is loadable."""
    try:
        _get_model()
        return True
    except Exception:
        return False
