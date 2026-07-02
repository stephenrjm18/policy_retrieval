"""
sql_prompt.py — Prompt template for LLM-based SQL generation (fallback).
"""

SQL_GENERATION_PROMPT = """\
You are an SQLite expert.
Tables:
{schema}

Rules:
- doctor/physician/specialist queries -> doctors table
- hospital/clinic queries -> hospitals table
- ALWAYS use LIKE '%keyword%' for text searches (names, cities, addresses)
- NEVER use = for text columns
- limit results to 50
- return ONLY the SQL query, no explanation, no markdown

Question: {question}
SQL:"""
