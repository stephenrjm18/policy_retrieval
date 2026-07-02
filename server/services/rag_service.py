"""
services/rag_service.py — Complete RAG retrieval + generation pipeline.

Pipeline: rewrite -> dense FAISS retrieval -> sparse BM25 retrieval ->
merge + deduplicate -> cross-encoder reranking -> LLM generation

Module-level singletons for the FAISS index, BM25 engine, and memory
are loaded once at startup and reused for all queries.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from database.vector_store import index_exists, load_index, search
from prompts.rag_prompt import RAG_PROMPT_TEMPLATE, HYBRID_RAG_PROMPT_TEMPLATE
from retrieval.hybrid_search import HybridSearch
from retrieval.reranker import rerank
from services.embedding_service import embed_query
from utils.ai_engine import query_llm
from utils.config import RAG_MODEL, RAG_TOP_K_DENSE, RAG_TOP_K_RERANK, RAG_TOP_K_SPARSE
from utils.logger import get_logger
from utils.memory import Memory
from utils.query_rewriter import rewrite_query

logger = get_logger(__name__)

# Module-level singletons

_index     = None
_texts:    list[str]  = []
_metadata: list[dict] = []
_bm25:     Optional[HybridSearch] = None
_memory    = Memory(max_turns=5)
_model     = RAG_MODEL

def get_model() -> str:
    return _model

def set_model(name: str) -> None:
    global _model
    _model = name
    logger.info("RAG model changed to: %s", name)

def get_chunk_count() -> int:
    _ensure_loaded()
    return len(_texts)

def clear_memory() -> None:
    _memory.clear()
    logger.info("RAG conversation memory cleared.")

def _ensure_loaded() -> None:
    """Load the vector store from disk exactly once per process."""
    global _index, _texts, _metadata, _bm25

    if _index is not None:
        return

    if not index_exists():
        raise RuntimeError(
            "Vector store not found. Run:  python ingest/ingest.py"
        )

    logger.info("Loading vector store from disk…")
    _index, _texts, _metadata = load_index()
    _bm25 = HybridSearch(_texts, _metadata)
    logger.info("Vector store loaded — %d chunks ready.", len(_texts))

# Retrieval helper

def _retrieve_and_rerank(rewritten_query: str) -> tuple[list[str], list[dict], list[float]]:
    """
    Run hybrid retrieval (dense + sparse) and rerank results.
    Returns (top_docs, top_meta, top_scores).
    """
    # Dense retrieval (FAISS)
    query_vec = embed_query(rewritten_query)
    faiss_docs, faiss_meta = search(
        _index, query_vec, _texts, _metadata, top_k=RAG_TOP_K_DENSE
    )

    # Sparse retrieval (BM25)
    bm25_docs, bm25_meta = _bm25.bm25_search(rewritten_query, top_k=RAG_TOP_K_SPARSE)

    # Merge + deduplicate (preserve insertion order)
    seen: set[str] = set()
    combined_docs: list[str] = []
    combined_meta: list[dict] = []
    for doc, meta in list(zip(faiss_docs, faiss_meta)) + list(zip(bm25_docs, bm25_meta)):
        if doc not in seen:
            seen.add(doc)
            combined_docs.append(doc)
            combined_meta.append(meta)

    # Cross-encoder reranking
    ranked    = rerank(rewritten_query, combined_docs, combined_meta, top_k=RAG_TOP_K_RERANK)
    top_docs  = [r[0] for r in ranked]
    top_meta  = [r[1] for r in ranked]
    top_scores = [r[2] for r in ranked]

    return top_docs, top_meta, top_scores

# Main retrieval function

import re as _re

def _clean_answer(text: str, query: str) -> str:
    """
    Strip accidental prompt-format echoes from LLM output.
    Removes leading 'Question:...', 'Answer:', 'Official Response:',
    'Employee Query:' lines that the LLM sometimes echoes back.
    """
    # Remove leading blank lines
    text = text.strip()

    # Remove any leading line that looks like a prompt label
    echo_patterns = [
        r"^Employee Query:.*?\n",
        r"^Official (?:Policy )?Response:\s*",
        r"^Question:.*?\n",
        r"^Answer:\s*",
        r"^Q:\s*.*?\n",
        r"^A:\s*",
    ]
    for pat in echo_patterns:
        text = _re.sub(pat, "", text, flags=_re.IGNORECASE | _re.DOTALL)

    return text.strip()

def ask(query: str) -> dict:
    """
    Full RAG pipeline for a policy query.
    Returns dict with: answer, sources, retrieval_time_ms, llm_time_ms, chunks_found.
    """
    _ensure_loaded()

    # Step 1: Query rewriting (only when history exists)
    t_ret = time.perf_counter()
    rewritten = rewrite_query(query, _memory, model=_model)
    logger.debug("Rewritten query: %s", rewritten)

    # Step 2 + 3: Hybrid retrieval + reranking
    top_docs, top_meta, top_scores = _retrieve_and_rerank(rewritten)
    retrieval_ms = int((time.perf_counter() - t_ret) * 1000)

    context       = "\n\n".join(top_docs)
    history_text  = _memory.format()
    history_block = f"\nConversation History:\n{history_text}\n" if history_text else ""

    # Step 4: LLM generation
    prompt = RAG_PROMPT_TEMPLATE.format(
        context=context,
        history_block=history_block,
        query=query,
    )

    t_llm = time.perf_counter()
    answer = query_llm(prompt, model=_model)
    llm_ms = int((time.perf_counter() - t_llm) * 1000)

    # Strip any accidental prompt-format echoes from the LLM output
    answer = _clean_answer(answer, query)

    # Step 5: Store turn in memory
    _memory.add(query, answer)

    # Step 6: Build deduplicated sources
    seen_keys: set[tuple] = set()
    sources: list[dict] = []
    for meta, score in zip(top_meta, top_scores):
        key = (meta.get("source", ""), meta.get("page", 1))
        if key not in seen_keys:
            seen_keys.add(key)
            sources.append({
                "source":    meta.get("source", "HVF Policy"),
                "page":      meta.get("page", 1),
                "relevance": round(float(score), 3),
            })

    return {
        "answer":            answer,
        "sources":           sources,
        "retrieval_time_ms": retrieval_ms,
        "llm_time_ms":       llm_ms,
        "chunks_found":      len(top_docs),
    }

def retrieve_context(query: str) -> tuple[str, list[dict], list[float], int]:
    """
    Retrieve and rerank without LLM generation.
    Used by hybrid_service to get the RAG context portion only.
    Returns (context_str, metadata_list, scores_list, retrieval_ms).
    """
    _ensure_loaded()

    t_ret = time.perf_counter()
    rewritten = rewrite_query(query, _memory, model=_model)
    top_docs, top_meta, top_scores = _retrieve_and_rerank(rewritten)
    retrieval_ms = int((time.perf_counter() - t_ret) * 1000)

    context = "\n\n".join(top_docs)
    return context, top_meta, top_scores, retrieval_ms
