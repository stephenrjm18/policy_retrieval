"""
metrics.py — Lightweight timing helpers.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Generator


class Timer:
    """Context manager that records wall-clock elapsed milliseconds."""

    def __init__(self) -> None:
        self.elapsed_ms: int = 0
        self._start: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_) -> None:
        self.elapsed_ms = int((time.perf_counter() - self._start) * 1000)


@contextmanager
def timed() -> Generator[Timer, None, None]:
    t = Timer()
    with t:
        yield t
