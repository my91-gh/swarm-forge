"""LLMClient: Claude CLI backend with disk cache and call logging.

Calls `claude -p <prompt>` via subprocess so no ANTHROPIC_API_KEY is required —
only a Claude subscription with the `claude` CLI installed and logged in.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


_DATA_DIR = Path(".skillops_data")
DEFAULT_MODEL = "cli"   # sentinel: means "whatever `claude` is logged in as"


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class BudgetExceededError(RuntimeError):
    """Kept for API compatibility; never raised when using the CLI backend."""


# ---------------------------------------------------------------------------
# LLMResponse
# ---------------------------------------------------------------------------

@dataclass
class LLMResponse:
    text: str
    input_tokens: int   # always 0 — CLI does not expose token counts
    output_tokens: int  # always 0
    cost_usd: float     # always 0.0 — covered by subscription
    latency_ms: float
    cached: bool = False
    model: str = ""


# ---------------------------------------------------------------------------
# DiskCache
# ---------------------------------------------------------------------------

class DiskCache:
    """SHA256-keyed response cache stored on disk."""

    def __init__(self, cache_dir: Path) -> None:
        self._dir = cache_dir
        self._dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self._dir / key[:2] / f"{key}.json"

    def get(self, payload: Dict[str, Any]) -> Optional[str]:
        p = self._path(self._hash(payload))
        if p.exists():
            return json.loads(p.read_text())["text"]
        return None

    def set(self, payload: Dict[str, Any], text: str) -> None:
        p = self._path(self._hash(payload))
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"text": text}))

    @staticmethod
    def _hash(payload: Dict[str, Any]) -> str:
        canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(canonical.encode()).hexdigest()


# ---------------------------------------------------------------------------
# LLMClient
# ---------------------------------------------------------------------------

class LLMClient:
    """Claude CLI client with disk cache and call logging.

    Invokes `claude -p <prompt>` for each LLM call so no API key is needed.
    Multi-turn messages are flattened into a single prompt; the system prompt
    is prepended in an XML-style block that Claude naturally respects.

    Parameters
    ----------
    model : str
        Ignored — the CLI uses whichever model the logged-in session provides.
        Accepted for interface compatibility; stored as ``self._model``.
    data_dir : Path
        Directory for the disk cache and call log.
    use_cache : bool
        Whether to cache responses by prompt hash (avoids re-calling the CLI
        for identical inputs, e.g. during tests).
    cli_binary : str
        Name or path of the Claude CLI binary (default ``"claude"``).
    timeout : int
        Subprocess timeout in seconds (default 120).
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        data_dir: Path = _DATA_DIR,
        use_cache: bool = True,
        cli_binary: str = "claude",
        timeout: int = 120,
        # kept for drop-in compatibility with the old API-based signature
        api_key: Optional[str] = None,
        hard_cap_usd: float = 0.0,
        warn_usd: float = 0.0,
    ) -> None:
        self._model = model
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._use_cache = use_cache
        self._cli = cli_binary
        self._timeout = timeout
        self._cache = DiskCache(self._data_dir / "cache")
        self._call_log = self._data_dir / "llm_calls.jsonl"

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def call(
        self,
        messages: List[Dict[str, Any]],
        system: Optional[str] = None,
        max_tokens: int = 4096,     # ignored by CLI; kept for compatibility
        model: Optional[str] = None,
        temperature: float = 0.0,   # ignored by CLI; kept for compatibility
    ) -> LLMResponse:
        prompt = self._flatten(system, messages)

        cache_payload: Dict[str, Any] = {"prompt": prompt}
        if self._use_cache:
            cached = self._cache.get(cache_payload)
            if cached is not None:
                resp = LLMResponse(
                    text=cached,
                    input_tokens=0, output_tokens=0, cost_usd=0.0,
                    latency_ms=0.0, cached=True, model=self._model,
                )
                self._log(resp, 0.0, None)
                return resp

        t0 = time.time()
        try:
            result = subprocess.run(
                [self._cli, "-p", prompt],
                capture_output=True,
                text=True,
                timeout=self._timeout,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"`{self._cli}` exited {result.returncode}: {result.stderr.strip()}"
                )
            text = result.stdout.strip()
        except Exception as exc:
            latency_ms = (time.time() - t0) * 1000.0
            self._log(None, latency_ms, str(exc))
            raise

        latency_ms = (time.time() - t0) * 1000.0
        if self._use_cache:
            self._cache.set(cache_payload, text)

        resp = LLMResponse(
            text=text,
            input_tokens=0, output_tokens=0, cost_usd=0.0,
            latency_ms=latency_ms, cached=False, model=self._model,
        )
        self._log(resp, latency_ms, None)
        return resp

    def count_tokens(
        self,
        messages: List[Dict[str, Any]],
        system: Optional[str] = None,
        model: Optional[str] = None,
    ) -> int:
        """Rough character-based estimate (CLI does not expose token counts)."""
        prompt = self._flatten(system, messages)
        return len(prompt) // 4   # ~4 chars per token

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _flatten(system: Optional[str], messages: List[Dict[str, Any]]) -> str:
        """Serialize system prompt + message list into a single string."""
        parts: List[str] = []
        if system:
            parts.append(f"<system>\n{system}\n</system>")
        for m in messages:
            role = m.get("role", "user")
            content = str(m.get("content", ""))
            if role == "user":
                parts.append(content)
            elif role == "assistant":
                parts.append(f"[Assistant previous turn]:\n{content}")
        return "\n\n".join(parts)

    def _log(
        self,
        resp: Optional[LLMResponse],
        latency_ms: float,
        error: Optional[str],
    ) -> None:
        entry: Dict[str, Any] = {
            "ts": time.time(),
            "backend": "claude-cli",
            "latency_ms": round(latency_ms, 2),
        }
        if resp:
            entry["cached"] = resp.cached
        if error:
            entry["error"] = error
        with open(self._call_log, "a") as fh:
            fh.write(json.dumps(entry) + "\n")
