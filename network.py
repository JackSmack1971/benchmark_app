"""
network.py — Network I/O Module
────────────────────────────────
Dedicated strictly to HTTP interactions with the OpenRouter API.

Responsibilities:
  • Maintain a persistent `requests.Session()` with connection pooling
  • Mount `HTTPAdapter` with `urllib3.util.Retry` for transient error recovery
  • Fetch and parse available free models into typed `ModelInfo` objects
  • Execute streaming benchmarks with TTFT capture and cancellation support
  • Provide header construction with dynamically injected API keys

This module imports ONLY from `config` (internal) and stdlib/requests (external).
It never imports Gradio, Pandas, Plotly, or accesses global environment state.
"""

from __future__ import annotations

import json
import time
import threading
from typing import Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from config import (
    OPENROUTER_BASE,
    HEADERS_BASE,
    SESSION_CONNECT_TIMEOUT,
    SESSION_READ_TIMEOUT,
    STREAM_READ_TIMEOUT,
    RETRY_TOTAL,
    RETRY_BACKOFF_FACTOR,
    RETRY_STATUS_FORCELIST,
    BenchmarkResult,
    ModelInfo,
)


# ── Persistent Session Factory ──────────────────────────────────────────────

def _build_session() -> requests.Session:
    """
    Construct a requests.Session with:
      • Connection pooling (automatic via Session)
      • Retry on transient HTTP errors with exponential backoff
      • Explicit timeout defaults baked into the adapter
    """
    session = requests.Session()

    retry_strategy = Retry(
        total=RETRY_TOTAL,
        backoff_factor=RETRY_BACKOFF_FACTOR,
        status_forcelist=RETRY_STATUS_FORCELIST,
        allowed_methods=["GET", "POST"],
        raise_on_status=False,
    )

    adapter = HTTPAdapter(
        max_retries=retry_strategy,
        pool_connections=10,
        pool_maxsize=20,
    )

    session.mount("https://", adapter)
    session.mount("http://", adapter)

    return session


# Module-level singleton — reused across all requests for TCP socket pooling.
_session: requests.Session = _build_session()


def _get_headers(api_key: str) -> dict[str, str]:
    """Build request headers with the explicitly injected API key."""
    headers = HEADERS_BASE.copy()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key.strip()}"
    return headers


# ── Model Discovery ─────────────────────────────────────────────────────────

def fetch_free_models(api_key: str) -> list[ModelInfo]:
    """
    Query OpenRouter /models, filter to free text-capable models,
    and return typed ModelInfo objects sorted alphabetically.

    Args:
        api_key: The user's specific API key for this session.

    Returns:
        List of ModelInfo on success.

    Raises:
        RuntimeError: On network or parsing failure (caller must handle).
    """
    try:
        resp = _session.get(
            f"{OPENROUTER_BASE}/models",
            headers=_get_headers(api_key),
            timeout=(SESSION_CONNECT_TIMEOUT, SESSION_READ_TIMEOUT),
        )
        resp.raise_for_status()
        data = resp.json().get("data", [])
    except requests.exceptions.Timeout as exc:
        raise RuntimeError(
            f"Model fetch timed out after {SESSION_READ_TIMEOUT}s: {exc}"
        ) from exc
    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(f"Connection failed: {exc}") from exc
    except requests.exceptions.HTTPError as exc:
        raise RuntimeError(f"HTTP error {resp.status_code}: {exc}") from exc
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"Request failed: {exc}") from exc
    except (ValueError, KeyError) as exc:
        raise RuntimeError(f"Failed to parse model response: {exc}") from exc

    free_models: list[ModelInfo] = []

    for m in data:
        pricing = m.get("pricing", {})
        prompt_price = str(pricing.get("prompt", "1"))
        completion_price = str(pricing.get("completion", "1"))

        if prompt_price != "0" or completion_price != "0":
            continue

        modality = m.get("architecture", {}).get("modality", "")
        if "text" not in modality:
            continue

        free_models.append(ModelInfo.from_api_dict(m))

    free_models.sort(key=lambda x: x.name.lower())
    return free_models


# ── Streaming Benchmark Runner ──────────────────────────────────────────────

def run_single_benchmark(
    api_key: str,
    model_id: str,
    model_name: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    cancel_event: threading.Event,
    suite_label: str = "",
) -> BenchmarkResult:
    """
    Execute a single streaming benchmark against one model.

    Captures:
      • Total latency (wall-clock, request → last token)
      • Time To First Token (TTFT)
      • Token counts (API-reported or estimated)
      • Full response text

    Args:
        api_key: The user's specific API key.
        cancel_event: Session-scoped event to abort in-flight streams safely.

    Returns:
        BenchmarkResult with either valid metrics or an error field set.
    """
    payload: dict = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stream": True,
    }

    start_time: float = time.perf_counter()
    ttft: Optional[float] = None
    full_response: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0

    try:
        resp = _session.post(
            f"{OPENROUTER_BASE}/chat/completions",
            headers=_get_headers(api_key),
            json=payload,
            stream=True,
            timeout=(SESSION_CONNECT_TIMEOUT, STREAM_READ_TIMEOUT),
        )
        resp.raise_for_status()

        for raw_line in resp.iter_lines():
            # ── Cancellation check (per-chunk granularity) ──────────
            if cancel_event.is_set():
                resp.close()
                raise _CancelledError("Cancelled by user")

            if not raw_line:
                continue

            decoded: str = raw_line.decode("utf-8", errors="replace")
            if not decoded.startswith("data: "):
                continue

            data_str: str = decoded[6:].strip()
            if data_str == "[DONE]":
                break

            try:
                chunk: dict = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            # ── Extract content delta ───────────────────────────────
            choices = chunk.get("choices", [])
            if choices:
                delta = choices[0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    if ttft is None:
                        ttft = time.perf_counter() - start_time
                    full_response += content

            # ── Extract usage (often arrives in final chunk) ────────
            usage = chunk.get("usage", {})
            if usage:
                prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                completion_tokens = usage.get(
                    "completion_tokens", completion_tokens
                )

        end_time: float = time.perf_counter()
        latency: float = end_time - start_time

        # ── Fallback token estimation when API omits usage ──────────
        if completion_tokens == 0:
            completion_tokens = max(1, len(full_response.split()) * 4 // 3)
        if prompt_tokens == 0:
            prompt_tokens = max(1, len(prompt.split()) * 4 // 3)

        tokens_per_sec: float = (
            round(completion_tokens / latency, 1) if latency > 0 else 0.0
        )

        return BenchmarkResult(
            model_id=model_id,
            model_name=model_name,
            prompt=prompt,
            response=full_response.strip(),
            latency_sec=round(latency, 3),
            ttft_sec=round(ttft, 3) if ttft is not None else None,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            tokens_per_sec=tokens_per_sec,
            temperature=temperature,
            top_p=top_p,
            max_tokens_cfg=max_tokens,
            suite_label=suite_label,
        )

    except _CancelledError:
        end_time = time.perf_counter()
        return BenchmarkResult(
            model_id=model_id,
            model_name=model_name,
            prompt=prompt,
            response=full_response.strip(),
            latency_sec=round(end_time - start_time, 3),
            ttft_sec=round(ttft, 3) if ttft is not None else None,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            tokens_per_sec=0.0,
            temperature=temperature,
            top_p=top_p,
            max_tokens_cfg=max_tokens,
            error="Cancelled by user",
            suite_label=suite_label,
        )

    except requests.exceptions.Timeout as exc:
        end_time = time.perf_counter()
        return BenchmarkResult(
            model_id=model_id,
            model_name=model_name,
            prompt=prompt,
            response="",
            latency_sec=round(end_time - start_time, 3),
            ttft_sec=None,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            tokens_per_sec=0.0,
            temperature=temperature,
            top_p=top_p,
            max_tokens_cfg=max_tokens,
            error=f"Timeout: {exc}",
            suite_label=suite_label,
        )

    except requests.exceptions.ConnectionError as exc:
        end_time = time.perf_counter()
        return BenchmarkResult(
            model_id=model_id,
            model_name=model_name,
            prompt=prompt,
            response="",
            latency_sec=round(end_time - start_time, 3),
            ttft_sec=None,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            tokens_per_sec=0.0,
            temperature=temperature,
            top_p=top_p,
            max_tokens_cfg=max_tokens,
            error=f"Connection error: {exc}",
            suite_label=suite_label,
        )

    except Exception as exc:
        end_time = time.perf_counter()
        return BenchmarkResult(
            model_id=model_id,
            model_name=model_name,
            prompt=prompt,
            response="",
            latency_sec=round(end_time - start_time, 3),
            ttft_sec=None,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            tokens_per_sec=0.0,
            temperature=temperature,
            top_p=top_p,
            max_tokens_cfg=max_tokens,
            error=str(exc)[:200],
            suite_label=suite_label,
        )


# ── Internal Exception ──────────────────────────────────────────────────────

class _CancelledError(Exception):
    """Raised internally when the cancellation event fires mid-stream."""
    pass
