"""
services/hybrid_service.py — Parallel RAG + SQL execution and result merging.

For HYBRID_QUERY:
  1. Run RAG retrieval (context only, no generation yet)
  2. Run SQL execution in parallel
  3. Combine context + SQL summary into a single unified LLM prompt
  4. Return one coherent answer with both policy info and DB data
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import time

from prompts.rag_prompt import HYBRID_RAG_PROMPT_TEMPLATE
from services.rag_service import retrieve_context, _memory, get_model
from services.sql_service import execute_sql_query, get_sql_summary
from utils.ai_engine import query_llm
from utils.logger import get_logger

logger = get_logger(__name__)

_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)


def ask_hybrid(query: str) -> dict:
    """
    Execute RAG retrieval + SQL query in parallel, then merge with one LLM call.

    Returns unified dict matching QueryResponse schema.
    """
    t0 = time.perf_counter()

    # Parallel execution
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        rag_future = pool.submit(retrieve_context, query)
        sql_future = pool.submit(execute_sql_query, query)

        try:
            context, rag_meta, rag_scores, retrieval_ms = rag_future.result(timeout=120)
        except Exception as exc:
            logger.error("Hybrid RAG retrieval failed: %s", exc)
            context, rag_meta, rag_scores, retrieval_ms = "", [], [], 0

        try:
            sql_result = sql_future.result(timeout=120)
        except Exception as exc:
            logger.error("Hybrid SQL execution failed: %s", exc)
            sql_result = {"answer": "", "sql_results": [], "sql_time_ms": 0}

    sql_rows     = sql_result.get("sql_results", [])
    sql_time_ms  = sql_result.get("sql_time_ms", 0)
    sql_intent   = sql_result.get("intent")
    table        = sql_intent.entity_type if sql_intent else "hospitals"

    sql_summary  = get_sql_summary(sql_rows, table)

    # Unified LLM call
    history_text  = _memory.format()
    history_block = f"\nConversation History:\n{history_text}\n" if history_text else ""

    prompt = HYBRID_RAG_PROMPT_TEMPLATE.format(
        context       = context or "No policy context available.",
        sql_summary   = sql_summary,
        history_block = history_block,
        query         = query,
    )

    t_llm  = time.perf_counter()
    model  = get_model()
    answer = query_llm(prompt, model=model)
    llm_ms = int((time.perf_counter() - t_llm) * 1000)

    # Store in memory
    _memory.add(query, answer)

    # Build sources
    seen_keys: set[tuple] = set()
    sources: list[dict] = []
    for meta, score in zip(rag_meta, rag_scores):
        key = (meta.get("source", ""), meta.get("page", 1))
        if key not in seen_keys:
            seen_keys.add(key)
            sources.append({
                "source":    meta.get("source", "HVF Policy"),
                "page":      meta.get("page", 1),
                "relevance": round(float(score), 3),
            })

    total_ms = int((time.perf_counter() - t0) * 1000)
    logger.info(
        "Hybrid complete — retrieval=%dms  sql=%dms  llm=%dms  total=%dms",
        retrieval_ms, sql_time_ms, llm_ms, total_ms,
    )

    return {
        "answer":            answer,
        "sources":           sources,
        "sql_results":       sql_rows,
        "retrieval_time_ms": retrieval_ms,
        "sql_time_ms":       sql_time_ms,
        "llm_time_ms":       llm_ms,
        "chunks_found":      len(rag_meta),
    }
