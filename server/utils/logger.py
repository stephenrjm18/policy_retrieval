"""
logger.py — Structured logging setup for the unified assistant.

Call get_logger(__name__) in every module instead of print().
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from utils.config import LOGS_DIR

_FORMATTER = logging.Formatter(
    "%(asctime)s  %(levelname)-8s  %(name)-30s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

_configured = False


def _configure() -> None:
    global _configured
    if _configured:
        return

    root = logging.getLogger()
    root.setLevel(logging.DEBUG if os.environ.get("DEBUG", "").lower() == "true" else logging.INFO)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(_FORMATTER)
    ch.setLevel(logging.INFO)
    root.addHandler(ch)

    # File handler
    log_path = LOGS_DIR / "server.log"
    fh = logging.FileHandler(str(log_path), encoding="utf-8")
    fh.setFormatter(_FORMATTER)
    fh.setLevel(logging.DEBUG)
    root.addHandler(fh)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a named logger; configures root logger on first call."""
    _configure()
    return logging.getLogger(name)
