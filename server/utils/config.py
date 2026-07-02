"""
config.py — Centralised environment-driven configuration.

All runtime settings are read once from environment variables (or .env).
Import this module instead of reading os.environ directly anywhere else.
"""

from __future__ import annotations

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]   # unified_assistant/
SERVER_ROOT  = Path(__file__).resolve().parents[1]   # unified_assistant/server/
DATA_DIR     = PROJECT_ROOT / "data"
LOGS_DIR     = PROJECT_ROOT / "logs"
VECTOR_DIR   = PROJECT_ROOT / "vector_store"
MODELS_DIR   = PROJECT_ROOT / "models"

LOGS_DIR.mkdir(exist_ok=True)

# Ollama
OLLAMA_URL        = os.environ.get("OLLAMA_URL",        "http://127.0.0.1:11434")
ROUTER_MODEL      = os.environ.get("ROUTER_MODEL",      "qwen2.5:1.5b")    # lightweight router
SQL_MODEL         = os.environ.get("SQL_MODEL",         "qwen2.5-coder:7b") # SQL generation
RAG_MODEL         = os.environ.get("RAG_MODEL",         "qwen2.5:3b")       # RAG answering
INTENT_MODEL      = os.environ.get("INTENT_MODEL",      "qwen2.5:1.5b")    # intent extraction

# Database
HOSPITAL_DB    = str(DATA_DIR / "hospital.db")
HOSPITALS_CSV  = str(DATA_DIR / "hospitals.csv")
DOCTORS_CSV    = str(DATA_DIR / "doctors.csv")

# Vector store
VECTOR_INDEX_PATH = str(VECTOR_DIR / "index.faiss")
VECTOR_META_PATH  = str(VECTOR_DIR / "meta.pkl")

# Retrieval
RAG_TOP_K_DENSE   = int(os.environ.get("RAG_TOP_K_DENSE", 15))
RAG_TOP_K_SPARSE  = int(os.environ.get("RAG_TOP_K_SPARSE", 15))
RAG_TOP_K_RERANK  = int(os.environ.get("RAG_TOP_K_RERANK", 8))

# Cache
CACHE_ENABLED     = os.environ.get("CACHE_ENABLED", "true").lower() == "true"
CACHE_MAX_ENTRIES = int(os.environ.get("CACHE_MAX_ENTRIES", 500))
CACHE_SIM_THRESH  = float(os.environ.get("CACHE_SIM_THRESH", 0.98))

# App
SECRET_KEY    = os.environ.get("SECRET_KEY", "unified-hvf-assistant-secret")
APP_HOST      = os.environ.get("APP_HOST", "0.0.0.0")
APP_PORT      = int(os.environ.get("APP_PORT", 8000))
DEBUG         = os.environ.get("DEBUG", "false").lower() == "true"

# Cross-encoder model
RERANKER_LOCAL_PATH  = str(MODELS_DIR / "ms-marco-MiniLM-L-6-v2")
RERANKER_HUB_NAME    = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANKER_ENABLED     = os.environ.get("RERANKER_ENABLED", "true").lower() == "true"