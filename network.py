"""
network.py — Asynchronous Network I/O Module
─────────────────────────────────────────────
High-performance async HTTP interactions with the OpenRouter API.

Responsibilities:
  • Maintain a persistent `httpx.AsyncClient` for async socket multiplexing
  • Execute asynchronous fetch and parsing of free models
  • Stream remote LLM computations without blocking the main event loop
  • Implement native async exponential backoff for HTTP 429/5xx absorption
  • Respond to instantaneous cancellation flags

This module imports ONLY from `config` (internal), `httpx`, and `asyncio`.
It completely avoids synchronous OS-thread blocking operations.
"""

from __future__ import annotations

import json
import time
import asyncio
from typing import Optional

import httpx

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


# ── Persistent Async Client Factory ─────────────────────────────────────────

def _build_async_client() -> httpx.AsyncClient:
    """
    Construct a persistent HTTPX AsyncClient with:
      • Native TCP connection pooling
      • Strict timeout tuples mapped to connection vs. read phases
    """
    limits = httpx.Limits(max_connections=20, max_keepalive_connections=10)
    timeout = httpx.Timeout(SESSION_READ_TIMEOUT, connect=SESSION_CONNECT_TIMEOUT)
    
    return httpx.AsyncClient(limits=limits, timeout=timeout)


# Module-level singleton — multiplexes all network I/O over a single thread.
_client: httpx.AsyncClient = _build_async_client()


def _get_headers(api_key: str) -> dict[str, str]:
    """Build request headers with the explicitly injected API key."""
    headers = HEADERS_BASE.copy()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key.strip()}"
    return headers


# ── Model Discovery (Async) ─────────────────────────────────────────────────

async def fetch_free_models(api_key: str) -> list[ModelInfo]:
    """
    Asynchronously query OpenRouter /models and filter to free text models.
    """
    for attempt in range(RETRY_TOTAL + 1):
        try:
            resp = await _client.get(
                f"{OPENROUTER_BASE}/models",
                headers=_get_headers(api_key),
            )
            resp.raise_for_status()
            data = resp.json().get("data", [])
            break  # Success, exit retry loop
            
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code not in RETRY_STATUS_FORCELIST or attempt == RETRY_TOTAL:
                raise RuntimeError(f"HTTP error {exc.response.status_code}: {exc}") from exc
            await asyncio.sleep(RETRY_BACKOFF_FACTOR * (2 ** attempt))
            
        except httpx.TimeoutException as exc:
            if attempt == RETRY_TOTAL:
                raise RuntimeError(f"Model fetch timed out: {exc}") from exc
            await asyncio.sleep(RETRY_BACKOFF_FACTOR * (2 ** attempt))
            
        except httpx.RequestError as exc:
            if attempt == RETRY_TOTAL:
                raise RuntimeError(f"Connection failed: {exc}") from exc
            await asyncio.sleep(RETRY_BACKOFF_FACTOR * (2 ** attempt))
            
        except (ValueError, KeyError) as exc:
            raise RuntimeError(f"Failed to parse model response: {exc}") from exc

    free_models: list[ModelInfo] = []

    for m in data:
        pricing = m.get("pricing", {})
        if str(pricing.get("prompt", "1")) != "0" or str(pricing.get("completion", "1")) != "0":
            continue

        modality = m.get("architecture", {}).get("modality", "")
        if "text" not in modality:
            continue

        free_models.append(ModelInfo.from_api_dict(m))

    free_models.sort(key=lambda x: x.name.lower())
    return free_models


# ── Streaming Benchmark Runner (Async) ──────────────────────────────────────

async def run_single_benchmark(
    api_key: str,
    model_id: str,
    model_name: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    cancel_flag: list[bool],
    suite_label: str = "",
) -> BenchmarkResult:
    """
    Execute a single streaming benchmark asynchronously.
    Yields control to the event loop during network I/O waits.
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
    
    stream_timeout = httpx.Timeout(STREAM_READ_TIMEOUT, connect=SESSION_CONNECT_TIMEOUT)
    req = _client.build_request("POST", f"{OPENROUTER_BASE}/chat/completions", headers=_get_headers(api_key), json=payload)

    try:
        # ── Async Exponential Backoff Phase ──
        resp = None
        for attempt in range(RETRY_TOTAL + 1):
            if cancel_flag[0]:
                raise _CancelledError("Cancelled by user")
                
            try:
                resp = await _client.send(req, stream=True, timeout=stream_timeout)
                resp.raise_for_status()
                break  # Headers received successfully
                
            except httpx.HTTPStatusError as exc:
                if resp: await resp.aclose()
                if exc.response.status_code not in RETRY_STATUS_FORCELIST or attempt == RETRY_TOTAL:
                    raise
                await asyncio.sleep(RETRY_BACKOFF_FACTOR * (2 ** attempt))
                
            except (httpx.ConnectError, httpx.TimeoutException) as exc:
                if resp: await resp.aclose()
                if attempt == RETRY_TOTAL:
                    raise
                await asyncio.sleep(RETRY_BACKOFF_FACTOR * (2 ** attempt))

        # ── Async Stream Processing Phase ──
        if not resp:
            raise RuntimeError("Failed to obtain response object.")

        try:
            async for raw_line in resp.aiter_lines():
                if cancel_flag[0]:
                    raise _CancelledError("Cancelled by user")

                if not raw_line or not raw_line.startswith("data: "):
                    continue

                data_str: str = raw_line[6:].strip()
                if data_str == "[DONE]":
                    break

                try:
                    chunk: dict = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                choices = chunk.get("choices", [])
                if choices:
                    content = choices[0].get("delta", {}).get("content", "")
                    if content:
                        if ttft is None:
                            ttft = time.perf_counter() - start_time
                        full_response += content

                usage = chunk.get("usage", {})
                if usage:
                    prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                    completion_tokens = usage.get("completion_tokens", completion_tokens)
        finally:
            await resp.aclose()

        end_time: float = time.perf_counter()
        latency: float = end_time - start_time

        # Fallback estimations
        if completion_tokens == 0:
            completion_tokens = max(1, len(full_response.split()) * 4 // 3)
        if prompt_tokens == 0:
            prompt_tokens = max(1, len(prompt.split()) * 4 // 3)

        tokens_per_sec: float = round(completion_tokens / latency, 1) if latency > 0 else 0.0

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
        return BenchmarkResult(
            model_id=model_id, model_name=model_name, prompt=prompt, response=full_response.strip(),
            latency_sec=round(time.perf_counter() - start_time, 3), ttft_sec=round(ttft, 3) if ttft is not None else None,
            prompt_tokens=0, completion_tokens=0, total_tokens=0, tokens_per_sec=0.0,
            temperature=temperature, top_p=top_p, max_tokens_cfg=max_tokens, error="Cancelled by user", suite_label=suite_label,
        )

    except Exception as exc:
        return BenchmarkResult(
            model_id=model_id, model_name=model_name, prompt=prompt, response="",
            latency_sec=round(time.perf_counter() - start_time, 3), ttft_sec=None,
            prompt_tokens=0, completion_tokens=0, total_tokens=0, tokens_per_sec=0.0,
            temperature=temperature, top_p=top_p, max_tokens_cfg=max_tokens, error=str(exc)[:200], suite_label=suite_label,
        )


class _CancelledError(Exception):
    """Raised internally when the cancellation flag is toggled mid-stream."""
    pass
