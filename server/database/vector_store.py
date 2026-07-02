"""
vector_store.py — FAISS vector store with per-chunk metadata.

Paths for the index and metadata files come from config.py.
"""

from __future__ import annotations

import os
import pickle

import faiss
import numpy as np

from utils.config import VECTOR_INDEX_PATH, VECTOR_META_PATH

_MIN_VALID_BYTES = 100

# Write

def create_index(embeddings: np.ndarray) -> faiss.Index:
    """Build a flat L2 FAISS index from an array of embeddings."""
    dim   = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings, dtype=np.float32))
    return index

def save_index(
    index: faiss.Index, texts: list[str], metadata: list[dict]
) -> None:
    """Persist the FAISS index and metadata to disk."""
    os.makedirs(os.path.dirname(VECTOR_INDEX_PATH), exist_ok=True)
    faiss.write_index(index, VECTOR_INDEX_PATH)
    with open(VECTOR_META_PATH, "wb") as f:
        pickle.dump({"texts": texts, "meta": metadata}, f)

# Read

def index_exists() -> bool:
    """Return True only when both files exist and are non-trivially sized."""
    for path in (VECTOR_INDEX_PATH, VECTOR_META_PATH):
        if not os.path.isfile(path):
            return False
        if os.path.getsize(path) < _MIN_VALID_BYTES:
            return False
    return True

def load_index() -> tuple[faiss.Index, list[str], list[dict]]:
    """Load and return (index, texts, metadata). Raises if files are missing."""
    if not index_exists():
        raise FileNotFoundError(
            "Vector store not found. Run:  python ingest/ingest.py"
        )
    index = faiss.read_index(VECTOR_INDEX_PATH)
    with open(VECTOR_META_PATH, "rb") as f:
        data = pickle.load(f)
    return index, data["texts"], data["meta"]

def search(
    index: faiss.Index,
    query_vec: np.ndarray,
    texts: list[str],
    metadata: list[dict],
    top_k: int = 10,
) -> tuple[list[str], list[dict]]:
    """Return (texts, metadata) for the top-k nearest neighbours."""
    vec = np.array([query_vec], dtype=np.float32)
    distances, indices = index.search(vec, min(top_k, len(texts)))
    result_texts = [texts[i] for i in indices[0] if i < len(texts)]
    result_meta  = [metadata[i] for i in indices[0] if i < len(metadata)]
    return result_texts, result_meta
