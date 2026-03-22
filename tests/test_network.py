"""
Tests for network.py, achieving 100% line coverage gate via respx mocking.
"""
import pytest
import asyncio
import httpx
import time
from config import OPENROUTER_BASE, ModelInfo
from network import (
    _get_headers,
    fetch_free_models,
    run_single_benchmark,
    _CancelledError
)

def test_get_headers():
    h = _get_headers("test-key")
    assert h["Authorization"] == "Bearer test-key"
    h_empty = _get_headers("")
    assert "Authorization" not in h_empty

@pytest.mark.asyncio
async def test_fetch_free_models_success(respx_mock):
    respx_mock.get(f"{OPENROUTER_BASE}/models").respond(
        status_code=200,
        json={
            "data": [
                {
                    "id": "free-model-1",
                    "name": "Free text model",
                    "pricing": {"prompt": "0", "completion": "0"},
                    "architecture": {"modality": "text-to-text"}
                },
                {
                    "id": "paid-model",
                    "pricing": {"prompt": "0.1", "completion": "0.1"},
                    "architecture": {"modality": "text"}
                },
                {
                    "id": "free-image",
                    "pricing": {"prompt": "0", "completion": "0"},
                    "architecture": {"modality": "image"}
                }
            ]
        }
    )
    models = await fetch_free_models("key")
    assert len(models) == 1
    assert models[0].id == "free-model-1"

@pytest.mark.asyncio
async def test_fetch_free_models_retry_status(respx_mock, monkeypatch):
    import network
    monkeypatch.setattr(network, "RETRY_BACKOFF_FACTOR", 0.01)
    route = respx_mock.get(f"{OPENROUTER_BASE}/models")
    route.side_effect = [
        httpx.Response(502),
        httpx.Response(200, json={"data": []})
    ]
    models = await fetch_free_models("key")
    assert len(models) == 0
    assert route.call_count == 2

@pytest.mark.asyncio
async def test_fetch_free_models_failure(respx_mock, monkeypatch):
    import network
    monkeypatch.setattr(network, "RETRY_BACKOFF_FACTOR", 0.01)
    monkeypatch.setattr(network, "RETRY_TOTAL", 1)
    respx_mock.get(f"{OPENROUTER_BASE}/models").respond(status_code=502)
    with pytest.raises(RuntimeError, match="HTTP error"):
        await fetch_free_models("key")

@pytest.mark.asyncio
async def test_run_single_benchmark_success(respx_mock):
    # Mocking the streaming endpoint
    def stream_response(request):
        chunks = [
            'data: {"choices": [{"delta": {"content": "Hello"}}]}\n',
            'data: {"choices": [{"delta": {"content": " world"}}]}\n',
            'data: {"usage": {"prompt_tokens": 10, "completion_tokens": 5}}\n',
            'data: [DONE]\n'
        ]
        return httpx.Response(
            200,
            headers={'content-type': 'text/event-stream'},
            text="".join(chunks)
        )
    
    route = respx_mock.post(f"{OPENROUTER_BASE}/chat/completions").mock(side_effect=stream_response)
    
    cancel_flag = [False]
    result = await run_single_benchmark(
        "key", "m-id", "m-name", "prompt", 10, 0.7, 1.0, cancel_flag, "label"
    )
    assert not result.is_error
    assert result.response == "Hello world"
    assert result.completion_tokens == 5
    assert result.latency_sec > 0

@pytest.mark.asyncio
async def test_run_single_benchmark_cancellation(respx_mock):
    # Let's cancel before request even fires
    cancel_flag = [True]
    result = await run_single_benchmark(
        "key", "m", "n", "p", 10, 0.7, 1.0, cancel_flag
    )
    assert result.is_error
    assert result.error == "Cancelled by user"

    # Now cancel mid-stream
    cancel_flag = [False]
    def slow_stream(request):
        # We will yield one chunk then cancel
        cancel_flag[0] = True
        return httpx.Response(200, text='data: {"choices": [{"delta": {"content": "H"}}]}\n')
    
    respx_mock.post(f"{OPENROUTER_BASE}/chat/completions").mock(side_effect=slow_stream)
    result = await run_single_benchmark("key", "m", "n", "p", 10, 0.7, 1.0, cancel_flag)
    assert result.is_error
    assert result.error == "Cancelled by user"

@pytest.mark.asyncio
async def test_run_single_benchmark_http_error(respx_mock, monkeypatch):
    import network
    monkeypatch.setattr(network, "RETRY_BACKOFF_FACTOR", 0.01)
    monkeypatch.setattr(network, "RETRY_TOTAL", 0)
    respx_mock.post(f"{OPENROUTER_BASE}/chat/completions").respond(400, text="Bad Request")
    cancel_flag = [False]
    result = await run_single_benchmark("key", "m", "n", "p", 10, 0.7, 1.0, cancel_flag)
    assert result.is_error
    assert "400" in result.error
