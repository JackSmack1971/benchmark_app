"""
conftest.py — Shared pytest fixtures for benchmark_app tests.
"""

import pytest
from config import BenchmarkResult, ModelInfo, LeaderboardRow


@pytest.fixture
def sample_result() -> BenchmarkResult:
    """A valid, successful BenchmarkResult."""
    return BenchmarkResult(
        model_id="openai/gpt-4o-mini",
        model_name="GPT-4o Mini",
        prompt="What is 2+2?",
        response="4",
        latency_sec=1.5,
        ttft_sec=0.3,
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
        tokens_per_sec=3.33,
    )


@pytest.fixture
def error_result() -> BenchmarkResult:
    """A BenchmarkResult representing a failed run."""
    return BenchmarkResult(
        model_id="openai/gpt-4o-mini",
        model_name="GPT-4o Mini",
        prompt="What is 2+2?",
        response="",
        latency_sec=0.0,
        ttft_sec=None,
        prompt_tokens=0,
        completion_tokens=0,
        total_tokens=0,
        tokens_per_sec=0.0,
        error="HTTP 429: Rate limit exceeded",
    )


@pytest.fixture
def sample_model_info() -> ModelInfo:
    """A valid ModelInfo built via from_api_dict."""
    return ModelInfo.from_api_dict({
        "id": "google/gemma-3-27b-it:free",
        "name": "Gemma 3 27B",
        "context_length": 131072,
        "architecture": {"modality": "text->text"},
        "top_provider": {"max_completion_tokens": 8192},
        "description": "Google's Gemma 3 27B instruction-tuned model.",
    })


@pytest.fixture
def sample_leaderboard_row() -> LeaderboardRow:
    """A single LeaderboardRow with realistic values."""
    return LeaderboardRow(
        model="GPT-4o Mini",
        model_id="openai/gpt-4o-mini",
        avg_lat=1.5,
        std_lat=0.2,
        avg_ttft=0.3,
        avg_tps=3.33,
        std_tps=0.1,
        cv_tps=3.0,
        avg_tokens=5,
        errors=0,
        runs=3,
        error_rate=0.0,
    )
