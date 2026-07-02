"""
services/sql_service.py — Complete SQL retrieval pipeline.

Pipeline: intent_extractor -> smart_sql_builder -> hospital_db.run_sql -> formatter

Also provides a raw SQL fallback via the LLM sql_generator for complex
queries that the smart builder cannot handle.
"""

from __future__ import annotations

import re
import time

from database.hospital_db import get_compact_schema, run_sql
from formatter import format_rows, format_error, format_sql_error
from intent_extractor import IntentResult, extract_intent
from prompts.sql_prompt import SQL_GENERATION_PROMPT
from smart_sql_builder import RouteResult, build_sql
from utils.ai_engine import query_llm
from utils.config import SQL_MODEL
from utils.logger import get_logger

logger = get_logger(__name__)

# SQL safety guard

_BLOCKED = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE|REPLACE|MERGE|EXEC|EXECUTE|GRANT|REVOKE)\b",
    re.IGNORECASE,
)

def _validate_sql(sql: str) -> None:
    if _BLOCKED.search(sql):
        raise ValueError("Blocked: non-SELECT keyword detected")
    if not sql.strip().upper().startswith("SELECT"):
        raise ValueError("Query does not start with SELECT")

def _clean_llm_sql(raw: str) -> str:
    raw = raw.strip()
    raw = re.sub(r"```(?:sql)?", "", raw, flags=re.IGNORECASE)
    raw = raw.strip("`").strip()
    return raw.split(";")[0].strip()

# LLM SQL generation fallback

def _generate_sql_via_llm(query: str) -> RouteResult | None:
    """Attempt LLM-based SQL generation. Returns None on failure."""
    schema = get_compact_schema()
    prompt = SQL_GENERATION_PROMPT.format(schema=schema, question=query)

    for attempt in (1, 2):
        try:
            raw = query_llm(prompt, model=SQL_MODEL, temperature=0)
            sql = _clean_llm_sql(str(raw))
            _validate_sql(sql)

            table = "doctors" if "FROM DOCTORS" in sql.upper() or "JOIN DOCTORS" in sql.upper() else "hospitals"
            logger.info("LLM SQL (attempt %d): %s", attempt, sql[:120])

            return RouteResult(
                sql=sql, params=(), table=table,
                strategy=f"llm_fallback:attempt{attempt}"
            )
        except Exception as exc:
            logger.warning("LLM SQL attempt %d failed: %s", attempt, exc)

    return None

# Main SQL execution function

def execute_sql_query(query: str) -> dict:
    """
    Full SQL pipeline: extract intent -> build SQL -> execute -> format.

    Returns dict with:
      answer:       formatted text response
      sql_results:  raw list of dicts for frontend table rendering
      sql_time_ms:  total SQL pipeline time
    """
    t0 = time.perf_counter()

    # Step 1: Extract intent
    intent = extract_intent(query)

    # Step 2: Handle trivial general queries
    if not intent.needs_sql and intent.fallback_text:
        return {
            "answer":      intent.fallback_text,
            "sql_results": [],
            "sql_time_ms": int((time.perf_counter() - t0) * 1000),
            "intent":      intent,
        }

    # Step 3: Build parameterised SQL from intent
    route_result = build_sql(intent)
    logger.info("SQL strategy: %s | sql=%.120s", route_result.strategy, route_result.sql)

    # Step 4: Execute
    try:
        rows = run_sql(route_result.sql, route_result.params)
    except Exception as exc:
        logger.error("SQL execution failed: %s | sql=%s", exc, route_result.sql)

        # Attempt LLM fallback
        llm_route = _generate_sql_via_llm(query)
        if llm_route:
            try:
                rows = run_sql(llm_route.sql, llm_route.params)
                route_result = llm_route
            except Exception as exc2:
                logger.error("LLM SQL fallback also failed: %s", exc2)
                return {
                    "answer":      format_error(str(exc)),
                    "sql_results": [],
                    "sql_time_ms": int((time.perf_counter() - t0) * 1000),
                    "intent":      intent,
                }
        else:
            return {
                "answer":      format_error(str(exc)),
                "sql_results": [],
                "sql_time_ms": int((time.perf_counter() - t0) * 1000),
                "intent":      intent,
            }

    # Step 5: Format
    answer = format_rows(rows, route_result.table, query, intent)

    return {
        "answer":      answer,
        "sql_results": rows,
        "sql_time_ms": int((time.perf_counter() - t0) * 1000),
        "intent":      intent,
    }

def get_sql_summary(rows: list[dict], table: str, max_rows: int = 10) -> str:
    """
    Generate a brief text summary of SQL results for use in hybrid prompts.
    """
    if not rows:
        return "No database records found."

    total = len(rows)
    shown = rows[:max_rows]

    if table == "doctors" or (shown and "doctor_name" in shown[0]):
        items = [r.get("doctor_name", "—") for r in shown]
        label = "doctors"
    elif table == "hospitals" or (shown and "hospital_name" in shown[0]):
        items = [f"{r.get('hospital_name', '—')} ({r.get('city', '')})" for r in shown]
        label = "hospitals"
    else:
        items = [str(list(r.values())[:2]) for r in shown]
        label = "records"

    summary = f"Found {total} {label}: " + ", ".join(items)
    if total > max_rows:
        summary += f" ... and {total - max_rows} more."
    return summary
