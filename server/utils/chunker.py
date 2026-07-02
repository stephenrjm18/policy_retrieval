"""
utils/chunker.py — Splits document text into overlapping chunks.
"""

from __future__ import annotations


def chunk_with_metadata(
    docs: list[dict],
    chunk_size: int = 500,
    overlap: int = 100,
) -> tuple[list[str], list[dict]]:
    """
    Chunk documents into overlapping text windows.

    docs entries require at minimum: {text, source}
    Optional: {category, page}

    Returns (chunks, chunk_meta) where chunk_meta has {source, page, category}.
    """
    chunks: list[str] = []
    chunk_meta: list[dict] = []

    for doc in docs:
        text = doc.get("text", "").strip()
        if not text:
            continue

        meta = {
            "source":   doc.get("source", "unknown"),
            "page":     doc.get("page", 1),
            "category": doc.get("category", "general"),
        }

        start = 0
        while start < len(text):
            end   = start + chunk_size
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
                chunk_meta.append(meta)
            start += chunk_size - overlap

    return chunks, chunk_meta
