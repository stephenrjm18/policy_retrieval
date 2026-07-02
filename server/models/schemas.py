"""
schemas.py — Typed Pydantic schemas for all API request/response models.

All API endpoints use these schemas for validation and serialization.
"""

from __future__ import annotations

from typing import Any, List, Optional
from pydantic import BaseModel, Field


# Request schemas

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000, description="User question")


class SetModelRequest(BaseModel):
    model_name: str = Field(..., min_length=1)


# Nested response types

class SourceItem(BaseModel):
    source:    str
    page:      int
    relevance: float


# SqlRow removed — sql_results is typed as List[dict] directly, no wrapper needed


# Main unified response

class QueryResponse(BaseModel):
    """
    Unified response format for ALL query types.

    The frontend only requires: answer, sources, retrieval_time_ms,
    llm_time_ms, chunks_found — all other fields are bonus metadata.
    """
    query_type:         str              # POLICY_QUERY | HOSPITAL_QUERY | DOCTOR_QUERY | HYBRID_QUERY | GENERAL_QUERY
    answer:             str
    sources:            List[SourceItem] = []
    sql_results:        List[dict]       = []
    retrieval_time_ms:  int              = 0
    sql_time_ms:        int              = 0
    llm_time_ms:        int              = 0
    chunks_found:       int              = 0
    cached:             bool             = False


# Health response

class HealthResponse(BaseModel):
    status:           str
    vector_db_chunks: Optional[int]
    llm_model:        str
    ollama_models:    List[str]
    index_ready:      bool
    db_ready:         bool


# Debug response

class IntentDebugResponse(BaseModel):
    query:             str
    route:             str
    entity_type:       str
    intent:            str
    filters:           dict
    needs_sql:         bool
    extraction_method: str
    generated_sql:     Optional[str]
    sql_params:        Optional[list]
    strategy:          Optional[str]
