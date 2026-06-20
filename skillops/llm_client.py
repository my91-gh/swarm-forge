"""LLMClient: Anthropic Claude interface with budget controls and caching."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import anthropic

# ---------------------------------------------------------------------------
# Pricing table: USD per million tokens
# ---------------------------------------------------------------------------

PRICE_TABLE_USD_PER_MTOK: Dict[str, Dict[str, float]] = {
    "claude-opus-4-8":   {"input": 5.00, "output": 25.00},
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
    "claude-haiku-4-5":  {"input": 1.00, "output":  5.00},
    "stub-zero":         {"input": 0.00, "output":  0.00},
}

DEFAULT_MODEL = "claude-opus-4-8"
_DATA_DIR = Path(".skillops_data")
_HARD_CAP_USD = 100.0
_WARN_USD = 80.0


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class BudgetExceededError(RuntimeError):
    """Raised when cumulative spend reaches or exceeds the hard cap."""


# ---------------------------------------------------------------------------
# LLMResponse
# ---------------------------------------------------------------------------

@dataclass
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
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
# BudgetTracker
# ---------------------------------------------------------------------------

class BudgetTracker:
    """File-locked JSON state tracking cumulative costs across processes."""

    def __init__(self, state_path: Path) -> None:
        self._path = state_path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if not self._path.exists():
            self._path.write_text(json.dumps({"total_usd": 0.0, "calls": 0}))

    def add(self, cost: float) -> float:
        with open(self._path, "r+") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            state = json.load(fh)
            state["total_usd"] = round(state.get("total_usd", 0.0) + cost, 6)
            state["calls"] = state.get("calls", 0) + 1
            fh.seek(0)
            json.dump(state, fh)
            fh.truncate()
            total = state["total_usd"]
            fcntl.flock(fh, fcntl.LOCK_UN)
        return total

    def total(self) -> float:
        with open(self._path) as fh:
            fcntl.flock(fh, fcntl.LOCK_SH)
            val = json.load(fh).get("total_usd", 0.0)
            fcntl.flock(fh, fcntl.LOCK_UN)
        return val


# ---------------------------------------------------------------------------
# LLMClient
# ---------------------------------------------------------------------------

class LLMClient:
    """Anthropic Claude client with budget controls, disk cache, and call logging.

    Deliberately minimal: no fallback providers, no automatic retries, no
    SDK-side rate limiting. API errors are written to the call log and re-raised
    so the caller can decide what to do.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        data_dir: Path = _DATA_DIR,
        hard_cap_usd: float = _HARD_CAP_USD,
        warn_usd: float = _WARN_USD,
        use_cache: bool = True,
    ) -> None:
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self._model = model
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._hard_cap = hard_cap_usd
        self._warn = warn_usd
        self._use_cache = use_cache
        self._cache = DiskCache(self._data_dir / "cache")
        self._budget = BudgetTracker(self._data_dir / "budget_state.json")
        self._call_log = self._data_dir / "llm_calls.jsonl"

    def _price(self, model: str, input_tok: int, output_tok: int) -> float:
        rates = PRICE_TABLE_USD_PER_MTOK.get(model, {"input": 0.0, "output": 0.0})
        return (input_tok * rates["input"] + output_tok * rates["output"]) / 1_000_000

    def call(
        self,
        messages: List[Dict[str, Any]],
        system: Optional[str] = None,
        max_tokens: int = 4096,
        model: Optional[str] = None,
        temperature: float = 0.0,
    ) -> LLMResponse:
        model = model or self._model

        if model == "stub-zero":
            return LLMResponse(
                text="[stub]", input_tokens=0, output_tokens=0,
                cost_usd=0.0, latency_ms=0.0, cached=False, model=model,
            )

        total = self._budget.total()
        if total >= self._hard_cap:
            raise BudgetExceededError(
                f"Budget hard cap ${self._hard_cap} reached (spent ${total:.4f})"
            )
        if total >= self._warn:
            warnings.warn(
                f"SkillOps LLM spend ${total:.4f} approaching cap ${self._hard_cap}",
                stacklevel=2,
            )

        cache_payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if system:
            cache_payload["system"] = system

        if self._use_cache:
            cached_text = self._cache.get(cache_payload)
            if cached_text is not None:
                resp = LLMResponse(
                    text=cached_text,
                    input_tokens=0, output_tokens=0, cost_usd=0.0,
                    latency_ms=0.0, cached=True, model=model,
                )
                self._log(model, messages, resp, 0.0, None)
                return resp

        t0 = time.time()
        try:
            client = anthropic.Anthropic(api_key=self._api_key)
            kwargs: Dict[str, Any] = dict(
                model=model,
                max_tokens=max_tokens,
                messages=messages,
                temperature=temperature,
            )
            if system:
                kwargs["system"] = system

            raw = client.messages.create(**kwargs)
            text = raw.content[0].text
            input_tok = raw.usage.input_tokens
            output_tok = raw.usage.output_tokens
            cost = self._price(model, input_tok, output_tok)
            latency_ms = (time.time() - t0) * 1000.0

            self._budget.add(cost)
            if self._use_cache:
                self._cache.set(cache_payload, text)

            resp = LLMResponse(
                text=text,
                input_tokens=input_tok,
                output_tokens=output_tok,
                cost_usd=cost,
                latency_ms=latency_ms,
                cached=False,
                model=model,
            )
        except Exception as exc:
            latency_ms = (time.time() - t0) * 1000.0
            self._log(model, messages, None, latency_ms, str(exc))
            raise

        self._log(model, messages, resp, resp.latency_ms, None)
        return resp

    def _log(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        resp: Optional[LLMResponse],
        latency_ms: float,
        error: Optional[str],
    ) -> None:
        entry: Dict[str, Any] = {
            "ts": time.time(),
            "model": model,
            "n_messages": len(messages),
            "latency_ms": round(latency_ms, 2),
        }
        if resp:
            entry.update({
                "input_tokens": resp.input_tokens,
                "output_tokens": resp.output_tokens,
                "cost_usd": resp.cost_usd,
                "cached": resp.cached,
            })
        if error:
            entry["error"] = error
        with open(self._call_log, "a") as fh:
            fh.write(json.dumps(entry) + "\n")

    def count_tokens(
        self,
        messages: List[Dict[str, Any]],
        system: Optional[str] = None,
        model: Optional[str] = None,
    ) -> int:
        """Count tokens for a message list without making a full completion call."""
        model = model or self._model
        if model == "stub-zero":
            return 0
        client = anthropic.Anthropic(api_key=self._api_key)
        kwargs: Dict[str, Any] = dict(model=model, messages=messages)
        if system:
            kwargs["system"] = system
        result = client.messages.count_tokens(**kwargs)
        return result.input_tokens
