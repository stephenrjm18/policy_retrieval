"""
services/response_service.py — Unified response normalisation.

Converts the raw dict output from any service into the QueryResponse schema.
Handles GENERAL_QUERY direct answers without any retrieval.
"""

from __future__ import annotations

import time

from models.schemas import QueryResponse, SourceItem
from utils.ai_engine import query_llm
from utils.config import RAG_MODEL
from utils.logger import get_logger

logger = get_logger(__name__)

# GENERAL_QUERY system prompt — restricted to HVF/CGHS domain guidance.
# If a user reaches GENERAL_QUERY it is almost certainly a greeting or a
# question the router couldn't classify. Steer them to the right path.
_GENERAL_SYSTEM = """\
You are the HVF Medical Assistant for Heavy Vehicle Factory (HVF), Chennai.
You help employees with CGHS/HVF medical reimbursement queries and hospital/doctor lookups.

If the user's message is a greeting or general question, introduce yourself briefly and
tell them what you can help with (policy questions, hospital search, doctor search).

If the question looks like it might be about medical policy, reimbursement, documents,
claims, or hospitals/doctors — say you can look that up and ask them to rephrase
as a specific question.

Keep responses concise (2-4 sentences max). Never make up policy details.
"""


def build_response(
    query_type: str,
    raw: dict,
    cached: bool = False,
) -> QueryResponse:
    """
    Normalise a raw service result dict into a QueryResponse.

    raw dict keys (all optional with defaults):
      answer, sources, sql_results, retrieval_time_ms, sql_time_ms,
      llm_time_ms, chunks_found
    """
    sources = [
        SourceItem(
            source    = s.get("source", ""),
            page      = int(s.get("page", 1)),
            relevance = float(s.get("relevance", 1.0)),
        )
        for s in raw.get("sources", [])
    ]

    return QueryResponse(
        query_type         = query_type,
        answer             = raw.get("answer", ""),
        sources            = sources,
        sql_results        = raw.get("sql_results", []),
        retrieval_time_ms  = raw.get("retrieval_time_ms", 0),
        sql_time_ms        = raw.get("sql_time_ms", 0),
        llm_time_ms        = raw.get("llm_time_ms", 0),
        chunks_found       = raw.get("chunks_found", 0),
        cached             = cached,
    )


def handle_general_query(query: str) -> dict:
    """
    Handle GENERAL_QUERY with a lightweight LLM call.
    Returns raw dict compatible with build_response().
    """
    prompt = f"{_GENERAL_SYSTEM}\n\nUser: {query}\nAssistant:"
    t_llm  = time.perf_counter()
    try:
        answer = query_llm(prompt, model=RAG_MODEL)
    except Exception as exc:
        logger.error("General query LLM failed: %s", exc)
        answer = (
            "Hello! I'm the HVF Medical Assistant. I can help you with:\n"
            "- Medical reimbursement policy and claim procedures\n"
            "- Finding CGHS empanelled hospitals and doctors\n\n"
            "Please ask a specific question to get started."
        )
    llm_ms = int((time.perf_counter() - t_llm) * 1000)

    return {
        "answer":            answer,
        "sources":           [],
        "sql_results":       [],
        "retrieval_time_ms": 0,
        "sql_time_ms":       0,
        "llm_time_ms":       llm_ms,
        "chunks_found":      0,
    }
