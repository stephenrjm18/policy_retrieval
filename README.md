# policy_retrieval
# HVF Medical Policy AI Assistant

## Architecture

```
unified_assistant/
├── server/
│   ├── app.py                  # FastAPI app / main entry point
│   ├── routers/
│   │   └── query_router.py     # LLM-based query routing
│   ├── services/
│   │   ├── rag_service.py      # RAG pipeline: rewrite -> dense -> BM25 -> rerank -> LLM
│   │   ├── sql_service.py      # Intent -> SQL -> execute -> format
│   │   ├── hybrid_service.py   # Parallel RAG + SQL, merged answer
│   │   ├── response_service.py # Normalises responses to a common shape
│   │   └── embedding_service.py
│   ├── retrieval/
│   │   ├── hybrid_search.py    # BM25 sparse retrieval
│   │   └── reranker.py         # Cross-encoder reranking
│   ├── database/
│   │   ├── hospital_db.py      # SQLite builder, indexes, FTS5, run_sql()
│   │   └── vector_store.py     # FAISS index save/load/search
│   ├── prompts/                # Prompt templates for router/RAG/SQL
│   ├── models/schemas.py       # Pydantic request/response schemas
│   ├── cache/semantic_cache.py # Embedding-based LRU cache
│   ├── utils/
│   │   ├── config.py           # Env-driven config
│   │   ├── logger.py
│   │   ├── metrics.py
│   │   ├── ai_engine.py        # Ollama HTTP client + subprocess fallback
│   │   ├── memory.py           # Rolling conversation memory
│   │   ├── query_rewriter.py   # History-aware query rewriting
│   │   ├── chunker.py
│   │   └── loader.py           # Multi-format document loader
│   ├── intent_extractor.py     # LLM-based SQL intent extraction
│   ├── smart_sql_builder.py    # Intent -> parameterised SQL
│   └── formatter.py            # DB rows -> readable text, no LLM call
├── ingest/ingest.py             # One-time RAG indexing script
├── static/index.html            # Frontend
├── data/                        # Source CSVs, policy docs, SQLite db
├── vector_store/                # FAISS index (built by ingest.py)
├── download_models.py           # Pre-download ML models for offline use
├── requirements.txt
└── .env.example
```

```

endpoints:

- `GET /health` — index/db readiness, chunk count, active model
- `POST /clear-memory` — reset conversation history and cache
- `POST /set-model?model_name=qwen2.5:7b` — swap the active RAG model
- `GET /hospitals` — all hospital records as JSON
- `POST /reload-csv` — re-sync CSVs into SQLite (rebuilds indexes + FTS5)
- `GET /debug/route?q=...` — inspect the routing decision for a query
- `GET /debug/intent?q=...` — inspect intent extraction + generated SQL

## Routing

| Route | Trigger | Pipeline |
|-------|---------|----------|
| `POLICY_QUERY` | Policy, claims, documents, procedures | RAG only |
| `HOSPITAL_QUERY` | Find hospitals, list facilities | SQL only |
| `DOCTOR_QUERY` | Find doctors, specialists | SQL only |
| `HYBRID_QUERY` | Needs both policy info and directory data | RAG + SQL in parallel |
| `GENERAL_QUERY` | Greetings, off-topic | Direct LLM answer |

Routing is done by `ROUTER_MODEL` via semantic classification, with a
heuristic rule-based fallback if the LLM is unavailable.

## Models

| Task | Model | Notes |
|------|-------|-------|
| Routing | `qwen2.5:1.5b` | Small and fast |
| Intent extraction | `qwen2.5:1.5b` | Structured SQL intent |
| SQL generation (fallback) | `qwen2.5-coder:7b` | Used when the smart builder can't handle a query |
| RAG / hybrid answering | `qwen2.5:3b` | Final answer generation |

