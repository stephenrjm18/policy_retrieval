"""
utils/loader.py — Loads data files into normalized document dicts.

Supports: .json, .pdf, .docx, .txt, .md
"""

from __future__ import annotations

import json
import os


def load_file(file_path: str) -> list[dict]:
    """Dispatch to the correct loader based on file extension."""
    ext  = os.path.splitext(file_path)[1].lower()
    name = os.path.basename(file_path)

    if ext == ".json":
        return _load_json(file_path, name)
    elif ext == ".pdf":
        return _load_pdf(file_path, name)
    elif ext == ".docx":
        return _load_docx(file_path, name)
    elif ext in (".txt", ".md"):
        return _load_text(file_path, name)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def _load_json(path: str, name: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    result = []
    for i, doc in enumerate(data):
        if isinstance(doc, str):
            result.append({"text": doc, "category": "general", "source": name, "page": i + 1})
            continue
        if not isinstance(doc, dict) or "text" not in doc:
            continue

        # Use the rich text field directly — it already contains all metadata
        text = doc["text"].strip()
        if not text:
            continue

        result.append({
            "text":     text,
            "category": doc.get("category", "general"),
            "source":   doc.get("source", name),
            "page":     doc.get("page", i + 1),
            "id":       doc.get("id", ""),   # preserved so ingest can detect pre-curated entries
        })
    return result


def _load_pdf(path: str, name: str) -> list[dict]:
    import fitz
    doc = fitz.open(path)
    result = []
    for i, page in enumerate(doc, start=1):
        text = page.get_text().strip()
        if text:
            result.append({"text": text, "category": "general", "source": name, "page": i})
    return result


def _load_docx(path: str, name: str) -> list[dict]:
    from docx import Document
    doc = Document(path)
    result = []
    for i, para in enumerate(doc.paragraphs, start=1):
        text = para.text.strip()
        if text:
            result.append({"text": text, "category": "general", "source": name, "page": i})
    return result


def _load_text(path: str, name: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
    return [
        {"text": p, "category": "general", "source": name, "page": i + 1}
        for i, p in enumerate(paragraphs)
    ]
