"""
utils/query_rewriter.py — Rewrites queries into standalone form using history.

Skipped entirely when there is no history (saves an LLM round-trip).
"""

from __future__ import annotations

from utils.ai_engine import query_llm
from utils.memory import Memory


def rewrite_query(query: str, memory: Memory, model: str = "qwen2.5:3b") -> str:
    """
    Return a self-contained rewrite of *query* using conversation history.
    Falls back to original query on empty output or error.
    """
    history = memory.format()

    if not history.strip():
        return query  # no history — skip extra LLM call

    prompt = (
        "Rewrite the user's question into a clear, self-contained question "
        "using the conversation history below. "
        "Return ONLY the rewritten question — no explanation, no preamble.\n\n"
        f"Conversation:\n{history}\n\n"
        f"User Question: {query}\n\n"
        "Rewritten Question:"
    )

    try:
        rewritten = query_llm(prompt, model=model).strip()
    except Exception:
        return query

    first_line = rewritten.splitlines()[0].strip() if rewritten else ""
    return first_line if first_line else query
