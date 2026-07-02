"""
ai_engine.py — Unified Ollama HTTP client.

Calls the Ollama REST API; falls back to subprocess CLI on connection failure.
"""

from __future__ import annotations

import json
import re
import subprocess
import urllib.error
import urllib.request

from utils.config import OLLAMA_URL

_GENERATE_URL = f"{OLLAMA_URL}/api/generate"
_TAGS_URL     = f"{OLLAMA_URL}/api/tags"
_DEFAULT_TIMEOUT = 180  # seconds

# Matches ANSI/VT100 escape sequences produced by `ollama run` in terminal mode.
# Standard sequences (ESC [ ... letter) and bare cursor codes without ESC prefix
# (e.g. [9D[K) that appear when subprocess captures a pseudo-terminal stream.
_ANSI_RE = re.compile(
    r'''(?:\x1B[@-Z\\-_]|[\x80-\x9A\x9C-\x9F]|(?:\x1B\[|\x9B)[0-?]*[ -/]*[@-~])'''
    r'''|\[\d*[A-Za-z]|\[\d+;\d+[A-Za-z]'''
)


def _strip_ansi(text: str) -> str:
    """Remove ANSI/VT100 escape sequences and collapse extra whitespace."""
    text = _ANSI_RE.sub(" ", text)
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()


def query_llm(prompt: str, model: str = "qwen2.5:3b", temperature: float = 0.0) -> str:
    """
    Send a prompt to an Ollama model and return the text response.
    Tries the HTTP API first; falls back to subprocess on connection error.
    """
    try:
        return _http_query(prompt, model, temperature)
    except (urllib.error.URLError, OSError):
        return _subprocess_query(prompt, model)


def _http_query(prompt: str, model: str, temperature: float) -> str:
    payload = json.dumps({
        "model":       model,
        "prompt":      prompt,
        "stream":      False,
        "options":     {"temperature": temperature},
    }).encode("utf-8")

    req = urllib.request.Request(
        _GENERATE_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=_DEFAULT_TIMEOUT) as resp:
        body = json.loads(resp.read().decode("utf-8"))
        text = body.get("response", "").strip()
        if not text:
            raise RuntimeError("Ollama returned an empty response.")
        return text


def _subprocess_query(prompt: str, model: str) -> str:
    """
    CLI fallback for when the HTTP API is unreachable.

    IMPORTANT: sets TERM=dumb and NO_COLOR=1 to prevent `ollama run` from
    emitting ANSI cursor-movement escape sequences (e.g. [9D[K) into stdout.
    Without these, the captured output contains raw VT100 codes that corrupt
    the answer text (doubled words, garbage characters).
    """
    import os
    env = os.environ.copy()
    env["TERM"]     = "dumb"   # disables ANSI cursor codes in ollama output
    env["NO_COLOR"] = "1"      # disables colour codes
    try:
        result = subprocess.run(
            ["ollama", "run", model, prompt],
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=_DEFAULT_TIMEOUT,
            env=env,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Ollama exited {result.returncode}: {result.stderr.strip()}")
        output = _strip_ansi(result.stdout).strip()
        if not output:
            raise RuntimeError("Ollama returned an empty response.")
        return output
    except FileNotFoundError:
        raise RuntimeError("Ollama is not installed or not on PATH.")
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"Ollama timed out after {_DEFAULT_TIMEOUT}s.")


def list_ollama_models() -> list[str]:
    """Return locally available Ollama model names; empty list if unreachable."""
    try:
        req = urllib.request.Request(_TAGS_URL, method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            return [m["name"] for m in body.get("models", [])]
    except Exception:
        pass
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")[1:]
            return [l.split()[0] for l in lines if l.strip()]
    except Exception:
        pass
    return []