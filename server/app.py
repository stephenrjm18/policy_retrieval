

from __future__ import annotations

import sys
import os

# Path setup: ensure server/ is on sys.path so relative imports work
_SERVER_DIR = os.path.dirname(os.path.abspath(__file__))
if _SERVER_DIR not in sys.path:
    sys.path.insert(0, _SERVER_DIR)

import time
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from cache.semantic_cache import get_cache
from database.hospital_db import build_hospital_db, get_all_hospitals, is_db_ready
from database.vector_store import index_exists
from models.schemas import (
    HealthResponse,
    IntentDebugResponse,
    QueryRequest,
    QueryResponse,
)
from routers.query_router import classify_query
from services import rag_service
from services.hybrid_service import ask_hybrid
from services.response_service import build_response, handle_general_query
from services.sql_service import execute_sql_query
from utils.ai_engine import list_ollama_models
from utils.config import APP_HOST, APP_PORT, DOCTORS_CSV, HOSPITAL_DB, HOSPITALS_CSV
from utils.logger import get_logger

logger = get_logger(__name__)


# Lifespan

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 60)
    logger.info("HVF Unified Medical AI Assistant — starting up")

    # Build / verify SQLite database
    if not is_db_ready():
        logger.info("Building SQLite hospital database from CSV…")
        try:
            build_hospital_db(HOSPITALS_CSV, DOCTORS_CSV, HOSPITAL_DB)
            logger.info("Hospital database ready.")
        except Exception as exc:
            logger.error("Failed to build hospital database: %s", exc)
    else:
        logger.info("Hospital database already present — skipping rebuild.")

    # Pre-load vector store
    if not index_exists():
        logger.warning(
            "Vector store not found! Run  python ingest/ingest.py  before sending policy queries."
        )
    else:
        try:
            count = rag_service.get_chunk_count()
            logger.info("Vector store loaded — %d chunks ready.", count)
        except Exception as exc:
            logger.error("Failed to pre-load vector store: %s", exc)

    # Pre-warm embedding model (used for cache + retrieval)
    try:
        from services.embedding_service import embed_query
        embed_query("warmup")
        logger.info("Embedding model warmed up.")
    except Exception as exc:
        logger.warning("Embedding model warmup failed: %s", exc)

    logger.info("Server ready.")
    logger.info("=" * 60)
    yield
    logger.info("Server shutting down.")


# App

app = FastAPI(
    title="HVF Unified Medical AI Assistant",
    description="AI assistant combining RAG-based policy retrieval with a SQL-backed hospital/doctor directory.",
    version="4.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request timing middleware

@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0       = time.perf_counter()
    response = await call_next(request)
    ms       = int((time.perf_counter() - t0) * 1000)
    logger.info("%s %s -> %d  (%dms)", request.method, request.url.path, response.status_code, ms)
    return response


# /health

@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health():
    """System status: index readiness, chunk count, available models."""
    vec_ready = index_exists()
    db_ready  = is_db_ready()
    chunks    = None

    if vec_ready:
        try:
            chunks = rag_service.get_chunk_count()
        except Exception:
            pass

    return {
        "status":           "ok",
        "vector_db_chunks": chunks,
        "llm_model":        rag_service.get_model(),
        "ollama_models":    list_ollama_models(),
        "index_ready":      vec_ready,
        "db_ready":         db_ready,
    }


# /query — main unified endpoint

@app.post("/query", response_model=QueryResponse, tags=["query"])
async def query_endpoint(req: QueryRequest):
    """
    Unified query endpoint. Routes to RAG, SQL, Hybrid, or General handler.
    Returns a consistent QueryResponse regardless of query type.
    """
    question = req.question.strip()
    logger.info("Query received: %.120s", question)

    # Semantic cache lookup
    cache = get_cache()
    try:
        from services.embedding_service import embed_query
        q_emb = embed_query(question)
        cached = cache.get(question, q_emb)
        if cached:
            logger.info("Cache hit — returning cached response.")
            cached["cached"] = True
            return QueryResponse.model_validate(cached)
    except Exception as exc:
        logger.warning("Cache lookup failed: %s", exc)
        q_emb = None

    # Semantic routing
    try:
        route = classify_query(question)
    except Exception as exc:
        logger.error("Routing failed: %s", exc)
        route = "POLICY_QUERY"   # safe fallback

    logger.info("Route: %s", route)

    # Dispatch to appropriate service
    try:
        if route == "POLICY_QUERY":
            raw = rag_service.ask(question)
            # No-info fallback: if RAG found nothing, give a helpful steer
            _ans = raw.get("answer", "").strip().lower()
            _NO_INFO_SIGNALS = (
                "i don't have that information",
                "i do not have that information",
                "not found in my knowledge base",
                "no information",
            )
            if any(_ans.startswith(s) for s in _NO_INFO_SIGNALS):
                logger.info("RAG returned no-info — enriching response with scope guidance.")
                raw["answer"] = (
                    raw.get("answer", "").strip()
                    + "\n\nFor this query you may also:\n"
                    "• Contact the **Medical Section** directly for case-specific guidance.\n"
                    "• Search for an empanelled **hospital or doctor** by name or location.\n"
                    "• Ask a related policy question — e.g. 'What documents are needed for "
                    "emergency reimbursement?' or 'What is the advance limit?'"
                )

        elif route in ("HOSPITAL_QUERY", "DOCTOR_QUERY"):
            raw = execute_sql_query(question)
            raw.setdefault("sources", [])
            raw.setdefault("retrieval_time_ms", 0)
            raw.setdefault("chunks_found", 0)
            raw.setdefault("llm_time_ms", 0)
            # SQL no-results fallback: if no DB records, try RAG as a safety net
            # This handles cases like "My hospital didn't mention CGHS codes" that
            # got mis-routed to HOSPITAL_QUERY — RAG may have the answer.
            _sql_ans = raw.get("answer", "").strip()
            _SQL_EMPTY_SIGNALS = ("no hospitals found", "no doctors found", "no records found")
            if any(s in _sql_ans.lower() for s in _SQL_EMPTY_SIGNALS):
                logger.info("SQL returned no results — trying RAG as fallback for: %s", question)
                try:
                    _rag_raw = rag_service.ask(question)
                    _rag_ans = _rag_raw.get("answer", "").strip().lower()
                    _no_info = any(_rag_ans.startswith(s) for s in (
                        "i don't have", "i do not have", "not found in my knowledge"))
                    if not _no_info and _rag_raw.get("answer", "").strip():
                        raw = _rag_raw
                        route = "POLICY_QUERY"
                        logger.info("RAG fallback succeeded — switching route to POLICY_QUERY.")
                except Exception as _rag_exc:
                    logger.warning("RAG fallback failed: %s", _rag_exc)

        elif route == "HYBRID_QUERY":
            raw = ask_hybrid(question)

        else:  # GENERAL_QUERY
            raw = handle_general_query(question)

    except RuntimeError as exc:
        # RAG not ready (vector store missing) — fall back to SQL or general
        if "Vector store" in str(exc):
            logger.warning("Vector store not ready — falling back to SQL for: %s", question)
            try:
                raw = execute_sql_query(question)
                raw.setdefault("sources", [])
                raw.setdefault("retrieval_time_ms", 0)
                raw.setdefault("chunks_found", 0)
                raw.setdefault("llm_time_ms", 0)
                route = "HOSPITAL_QUERY"
            except Exception as exc2:
                raise HTTPException(status_code=503, detail=str(exc)) from exc2
        else:
            raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        logger.exception("Unexpected error handling query: %s", question)
        raise HTTPException(status_code=500, detail="Internal server error. Check server logs.")

    # Build unified response
    response = build_response(query_type=route, raw=raw)

    # Cache the result
    # Never cache empty answers or LLM-timeout failures.
    # A valid cacheable response must have a non-empty answer AND the LLM must
    # have actually run (llm_time_ms > 0), except for SQL-only routes which
    # return answers without an LLM call.
    _answer = response.answer.strip()
    _ERROR_PREFIXES = (
        "i don't have that information",
        "i do not have that information",
        "not found in my knowledge base",
        "i'm sorry, i cannot",
        "internal server",
        "no hospitals found",
        "no doctors found",
        "no records found",
    )
    _answer_ok = bool(_answer) and not any(
        _answer.lower().startswith(p) for p in _ERROR_PREFIXES
    )
    _llm_ran = (
        response.llm_time_ms > 0
        or route in ("HOSPITAL_QUERY", "DOCTOR_QUERY")  # SQL-only; no LLM needed
    )
    if q_emb is not None and _answer_ok and _llm_ran:
        try:
            cache.set(question, q_emb, response.model_dump())
        except Exception as exc:
            logger.warning("Cache store failed: %s", exc)
    elif q_emb is not None:
        logger.info(
            "Cache SKIP — empty answer or LLM timeout. llm_time_ms=%d  preview='%.60s'",
            response.llm_time_ms, _answer[:60],
        )

    return response


# /clear-memory

@app.post("/clear-memory", tags=["rag"])
async def clear_memory():
    """Reset conversation memory and semantic cache."""
    rag_service.clear_memory()
    get_cache().clear()
    return {"status": "ok", "message": "Conversation memory and cache cleared."}


# /set-model

@app.post("/set-model", tags=["system"])
async def set_model(model_name: str):
    """Change the active RAG model at runtime."""
    if not model_name.strip():
        raise HTTPException(status_code=400, detail="model_name cannot be empty.")
    rag_service.set_model(model_name.strip())
    logger.info("RAG model switched to: %s", model_name)
    return {"status": "ok", "model": model_name}


# /hospitals

@app.get("/hospitals", tags=["data"])
async def list_hospitals():
    """All hospitals as JSON — consumed by the frontend summary table."""
    try:
        rows = get_all_hospitals()
        return {
            "hospitals": rows,
            "columns":   list(rows[0].keys()) if rows else [],
        }
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={"error": str(exc), "hospitals": [], "columns": []},
        )


# /reload-csv

@app.post("/reload-csv", tags=["data"])
async def reload_csv():
    """Re-sync CSV files -> SQLite (rebuilds indexes and FTS5)."""
    try:
        build_hospital_db(HOSPITALS_CSV, DOCTORS_CSV, HOSPITAL_DB)
        return {"message": "Database reloaded. Indexes and FTS5 rebuilt."}
    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": str(exc)})


# /debug/route

@app.get("/debug/route", tags=["debug"])
async def debug_route(q: str = ""):
    """Show semantic routing decision for a query."""
    if not q:
        raise HTTPException(status_code=400, detail="Provide ?q=your+query")
    route = classify_query(q)
    return {"query": q, "route": route}


# /debug/intent

@app.get("/debug/intent", tags=["debug"])
async def debug_intent(q: str = ""):
    """Show full intent extraction + SQL plan for a query."""
    if not q:
        raise HTTPException(status_code=400, detail="Provide ?q=your+query")

    from intent_extractor import extract_intent
    from smart_sql_builder import build_sql

    route  = classify_query(q)
    intent = extract_intent(q)
    sql_r  = build_sql(intent)

    return {
        "query":             q,
        "route":             route,
        "entity_type":       intent.entity_type,
        "intent":            intent.intent,
        "filters":           intent.filters,
        "needs_sql":         intent.needs_sql,
        "extraction_method": intent.extraction_method,
        "generated_sql":     sql_r.sql,
        "sql_params":        list(sql_r.params),
        "strategy":          sql_r.strategy,
    }


# Global error handler

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception on %s", request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred. Check server logs."},
    )


# Static files — MUST BE LAST

_static_dir = os.path.join(os.path.dirname(_SERVER_DIR), "static")
if os.path.isdir(_static_dir):
    app.mount("/", StaticFiles(directory=_static_dir, html=True), name="static")
else:
    logger.warning("static/ directory not found at %s — frontend not served.", _static_dir)


# Dev entry point

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host=APP_HOST,
        port=APP_PORT,
        reload=False,
        workers=1,
        log_level="info",
    )