
from __future__ import annotations

import logging
import os
import sys
import time

# Path bootstrap
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SERVER_ROOT  = os.path.join(PROJECT_ROOT, "server")
for path in (PROJECT_ROOT, SERVER_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

from utils.chunker import chunk_with_metadata
from utils.loader import load_file

# Change to project root so vector_store/ paths resolve correctly
os.chdir(PROJECT_ROOT)

from database.vector_store import create_index, save_index
from services.embedding_service import embed_texts

# Config
DATA_FOLDER = os.path.join(PROJECT_ROOT, "data")
CHUNK_SIZE  = 500
OVERLAP     = 100

SUPPORTED_EXTENSIONS = {".json", ".pdf", ".docx", ".txt", ".md"}

# JSON files with pre-curated entries should NOT be chunked further.
# They are identified by having a "category" field in their entries.
def _is_precurated(doc: dict) -> bool:
    """Returns True if doc came from a pre-curated JSON policy file."""
    return bool(doc.get("category") and doc.get("id"))
SKIP_EXTENSIONS      = {".csv", ".db"}   # SQL data — not for RAG

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger("ingest")


def ingest() -> None:
    t_start = time.perf_counter()

    logger.info("=" * 60)
    logger.info("HVF Unified Assistant — Data Ingestion")
    logger.info("Data folder : %s", DATA_FOLDER)
    logger.info("Chunk size  : %d chars  |  Overlap: %d chars", CHUNK_SIZE, OVERLAP)
    logger.info("=" * 60)

    # 1. Load all policy data files
    all_docs: list[dict] = []
    files = sorted(os.listdir(DATA_FOLDER))

    if not files:
        logger.error("No files found in %s — aborting.", DATA_FOLDER)
        sys.exit(1)

    for fname in files:
        ext = os.path.splitext(fname)[1].lower()
        if ext in SKIP_EXTENSIONS:
            logger.debug("Skipping (SQL data): %s", fname)
            continue
        if ext not in SUPPORTED_EXTENSIONS:
            logger.debug("Skipping (unsupported): %s", fname)
            continue

        path = os.path.join(DATA_FOLDER, fname)
        try:
            docs = load_file(path)
            all_docs.extend(docs)
            logger.info("  ✓ %-45s -> %d documents", fname, len(docs))
        except Exception as exc:
            logger.warning("  ✗ %-45s -> %s (skipped)", fname, exc)

    if not all_docs:
        logger.error(
            "No policy documents loaded from %s.\n"
            "Add .json, .pdf, .docx, .txt, or .md files containing policy text.",
            DATA_FOLDER,
        )
        sys.exit(1)

    logger.info("")
    logger.info("Total documents loaded : %d", len(all_docs))

    # 2. Chunk
    # JSON files are already pre-curated entries — do NOT chunk them further.
    # Only chunk PDF/DOCX/TXT files which produce large raw paragraphs.
    # This preserves complete numbered lists (e.g. all 11 inpatient documents).
    logger.info("Chunking documents (JSON entries preserved as-is)…")

    json_docs  = [d for d in all_docs if d.get("source", "").endswith(".json")
                  or any(d.get("source","").endswith(ext) for ext in (".json",))]
    # Check if source filename suggests pre-curated JSON
    json_docs  = [d for d in all_docs if _is_precurated(d)]
    other_docs = [d for d in all_docs if not _is_precurated(d)]

    # Pre-curated JSON: use as-is (each entry is already one chunk)
    json_chunks = [d["text"] for d in json_docs]
    json_meta   = [{"source": d.get("source","unknown"), "page": d.get("page",1),
                    "category": d.get("category","general")} for d in json_docs]

    # Everything else: chunk normally
    other_chunks, other_meta = chunk_with_metadata(
        other_docs, chunk_size=CHUNK_SIZE, overlap=OVERLAP
    )

    chunks     = json_chunks + other_chunks
    chunk_meta = json_meta   + other_meta

    logger.info("Total chunks           : %d  (json=%d  chunked=%d)",
                len(chunks), len(json_chunks), len(other_chunks))

    # 3. Embed
    logger.info("Embedding chunks (may take a minute on first run)…")
    t_embed    = time.perf_counter()
    embeddings = embed_texts(chunks)
    logger.info("Embedding done in %.1f s  |  dim=%d", time.perf_counter() - t_embed, embeddings.shape[1])

    # 4. Build FAISS index
    logger.info("Building FAISS index…")
    index = create_index(embeddings)

    # 5. Save
    logger.info("Saving to vector_store/…")
    save_index(index, chunks, chunk_meta)

    elapsed = time.perf_counter() - t_start
    logger.info("")
    logger.info("=" * 60)
    logger.info("✓ Ingestion complete!")
    logger.info("  Chunks indexed : %d", len(chunks))
    logger.info("  Time elapsed   : %.1f s", elapsed)
    logger.info("  Output         : vector_store/index.faiss + meta.pkl")
    logger.info("")
    logger.info("Start the server with:")
    logger.info("  uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 1")
    logger.info("=" * 60)


if __name__ == "__main__":
    ingest()
