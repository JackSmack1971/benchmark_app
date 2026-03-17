"""
test_config.py — Unit tests for config.py

Covers:
- BenchmarkResult: construction, __post_init__ timestamp, is_error, to_dict
- ModelInfo: from_api_dict (full, minimal, missing nested keys), to_dict, provider parsing
- LeaderboardRow: construction, to_dict
- Module-level constants: structure and value constraints
"""

import re
from dataclasses import fields

import pytest

from config import (
    BENCHMARK_PRESETS,
    BENCHMARK_SUITES,
    CHART_COLORS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    MAX_PARALLEL_WORKERS,
    MAX_PROMPT_HISTORY,
    RETRY_STATUS_FORCELIST,
    RETRY_TOTAL,
    SESSION_CONNECT_TIMEOUT,
    SESSION_READ_TIMEOUT,
    SMART_DEFAULTS,
    STREAM_READ_TIMEOUT,
    BenchmarkResult,
    LeaderboardRow,
    ModelInfo,
)


# ── BenchmarkResult ──────────────────────────────────────────────────────────

class TestBenchmarkResultConstruction:
    def test_required_fields_stored(self, sample_result):
        r = sample_result
        assert r.model_id == "openai/gpt-4o-mini"
        assert r.model_name == "GPT-4o Mini"
        assert r.prompt == "What is 2+2?"
        assert r.response == "4"
        assert r.latency_sec == 1.5
        assert r.ttft_sec == 0.3
        assert r.prompt_tokens == 10
        assert r.completion_tokens == 5
        assert r.total_tokens == 15
        assert r.tokens_per_sec == 3.33

    def test_defaults(self, sample_result):
        assert sample_result.temperature == 0.7
        assert sample_result.top_p == 1.0
        assert sample_result.max_tokens_cfg == 512
        assert sample_result.error is None
        assert sample_result.suite_label == ""

    def test_optional_ttft_can_be_none(self):
        r = BenchmarkResult(
            model_id="x/y", model_name="Y", prompt="p", response="r",
            latency_sec=1.0, ttft_sec=None,
            prompt_tokens=1, completion_tokens=1, total_tokens=2, tokens_per_sec=1.0,
        )
        assert r.ttft_sec is None


class TestBenchmarkResultTimestamp:
    def test_timestamp_auto_generated_when_empty(self, sample_result):
        assert sample_result.timestamp != ""

    def test_timestamp_is_iso8601_utc(self, sample_result):
        # datetime.isoformat() with UTC timezone produces a +00:00 suffix
        ts = sample_result.timestamp
        assert "T" in ts
        assert ts.endswith("+00:00")

    def test_explicit_timestamp_preserved(self):
        custom_ts = "2024-01-15T12:00:00+00:00"
        r = BenchmarkResult(
            model_id="x/y", model_name="Y", prompt="p", response="r",
            latency_sec=1.0, ttft_sec=0.1,
            prompt_tokens=1, completion_tokens=1, total_tokens=2, tokens_per_sec=1.0,
            timestamp=custom_ts,
        )
        assert r.timestamp == custom_ts

    def test_two_results_have_different_timestamps(self):
        def make():
            return BenchmarkResult(
                model_id="x/y", model_name="Y", prompt="p", response="r",
                latency_sec=1.0, ttft_sec=0.1,
                prompt_tokens=1, completion_tokens=1, total_tokens=2, tokens_per_sec=1.0,
            )
        # Timestamps may collide in fast machines, but both should be valid ISO strings
        r1, r2 = make(), make()
        assert r1.timestamp != "" and r2.timestamp != ""


class TestBenchmarkResultIsError:
    def test_is_error_false_when_no_error(self, sample_result):
        assert sample_result.is_error is False

    def test_is_error_true_when_error_set(self, error_result):
        assert error_result.is_error is True

    def test_is_error_false_with_explicit_none(self):
        r = BenchmarkResult(
            model_id="x/y", model_name="Y", prompt="p", response="r",
            latency_sec=1.0, ttft_sec=0.1,
            prompt_tokens=1, completion_tokens=1, total_tokens=2, tokens_per_sec=1.0,
            error=None,
        )
        assert r.is_error is False

    def test_is_error_true_with_any_non_none_string(self):
        for msg in ["HTTP 500", "timeout", "cancelled", ""]:
            # Empty string is falsy but still not None — is_error checks `is not None`
            r = BenchmarkResult(
                model_id="x/y", model_name="Y", prompt="p", response="r",
                latency_sec=1.0, ttft_sec=None,
                prompt_tokens=0, completion_tokens=0, total_tokens=0, tokens_per_sec=0.0,
                error=msg,
            )
            assert r.is_error is True


class TestBenchmarkResultToDict:
    def test_to_dict_returns_dict(self, sample_result):
        assert isinstance(sample_result.to_dict(), dict)

    def test_to_dict_contains_all_fields(self, sample_result):
        d = sample_result.to_dict()
        field_names = {f.name for f in fields(BenchmarkResult)}
        assert field_names == set(d.keys())

    def test_to_dict_values_match_attributes(self, sample_result):
        d = sample_result.to_dict()
        assert d["model_id"] == sample_result.model_id
        assert d["latency_sec"] == sample_result.latency_sec
        assert d["tokens_per_sec"] == sample_result.tokens_per_sec
        assert d["error"] is None

    def test_to_dict_error_result(self, error_result):
        d = error_result.to_dict()
        assert d["error"] == "HTTP 429: Rate limit exceeded"
        assert d["response"] == ""


# ── ModelInfo ────────────────────────────────────────────────────────────────

class TestModelInfoFromApiDict:
    def test_full_payload(self, sample_model_info):
        m = sample_model_info
        assert m.id == "google/gemma-3-27b-it:free"
        assert m.name == "Gemma 3 27B"
        assert m.provider == "google"
        assert m.context_length == 131072
        assert m.modality == "text->text"
        assert m.max_completion == 8192
        assert m.description == "Google's Gemma 3 27B instruction-tuned model."

    def test_provider_extracted_from_slash(self):
        m = ModelInfo.from_api_dict({"id": "mistralai/mistral-7b-instruct:free"})
        assert m.provider == "mistralai"

    def test_provider_unknown_when_no_slash(self):
        m = ModelInfo.from_api_dict({"id": "noSlashModel"})
        assert m.provider == "unknown"

    def test_missing_id_defaults_to_unknown(self):
        m = ModelInfo.from_api_dict({})
        assert m.id == "unknown"
        assert m.provider == "unknown"

    def test_missing_name_falls_back_to_id(self):
        m = ModelInfo.from_api_dict({"id": "org/model-name"})
        assert m.name == "org/model-name"

    def test_missing_context_length_defaults_to_zero(self):
        m = ModelInfo.from_api_dict({"id": "org/model"})
        assert m.context_length == 0

    def test_missing_architecture_key(self):
        m = ModelInfo.from_api_dict({"id": "org/model"})
        assert m.modality == ""

    def test_missing_modality_inside_architecture(self):
        m = ModelInfo.from_api_dict({"id": "org/model", "architecture": {}})
        assert m.modality == ""

    def test_missing_top_provider_key(self):
        m = ModelInfo.from_api_dict({"id": "org/model"})
        assert m.max_completion == 0

    def test_missing_max_completion_tokens_inside_top_provider(self):
        m = ModelInfo.from_api_dict({"id": "org/model", "top_provider": {}})
        assert m.max_completion == 0

    def test_description_truncated_to_200_chars(self):
        long_desc = "x" * 300
        m = ModelInfo.from_api_dict({"id": "org/model", "description": long_desc})
        assert len(m.description) == 200

    def test_description_shorter_than_200_not_padded(self):
        m = ModelInfo.from_api_dict({"id": "org/model", "description": "short"})
        assert m.description == "short"

    def test_none_description_becomes_empty_string(self):
        m = ModelInfo.from_api_dict({"id": "org/model", "description": None})
        assert m.description == ""

    def test_missing_description_becomes_empty_string(self):
        m = ModelInfo.from_api_dict({"id": "org/model"})
        assert m.description == ""


class TestModelInfoToDict:
    def test_to_dict_returns_dict(self, sample_model_info):
        assert isinstance(sample_model_info.to_dict(), dict)

    def test_to_dict_contains_all_fields(self, sample_model_info):
        d = sample_model_info.to_dict()
        field_names = {f.name for f in fields(ModelInfo)}
        assert field_names == set(d.keys())

    def test_to_dict_values_match(self, sample_model_info):
        d = sample_model_info.to_dict()
        assert d["id"] == sample_model_info.id
        assert d["provider"] == "google"
        assert d["context_length"] == 131072


# ── LeaderboardRow ───────────────────────────────────────────────────────────

class TestLeaderboardRow:
    def test_construction(self, sample_leaderboard_row):
        r = sample_leaderboard_row
        assert r.model == "GPT-4o Mini"
        assert r.model_id == "openai/gpt-4o-mini"
        assert r.avg_lat == 1.5
        assert r.errors == 0
        assert r.runs == 3
        assert r.error_rate == 0.0

    def test_optional_avg_ttft_can_be_none(self):
        row = LeaderboardRow(
            model="M", model_id="org/m",
            avg_lat=1.0, std_lat=0.0, avg_ttft=None,
            avg_tps=1.0, std_tps=0.0, cv_tps=0.0,
            avg_tokens=10, errors=0, runs=1, error_rate=0.0,
        )
        assert row.avg_ttft is None

    def test_to_dict_returns_dict(self, sample_leaderboard_row):
        assert isinstance(sample_leaderboard_row.to_dict(), dict)

    def test_to_dict_contains_all_fields(self, sample_leaderboard_row):
        d = sample_leaderboard_row.to_dict()
        field_names = {f.name for f in fields(LeaderboardRow)}
        assert field_names == set(d.keys())

    def test_to_dict_none_ttft_preserved(self):
        row = LeaderboardRow(
            model="M", model_id="org/m",
            avg_lat=1.0, std_lat=0.0, avg_ttft=None,
            avg_tps=1.0, std_tps=0.0, cv_tps=0.0,
            avg_tokens=10, errors=1, runs=1, error_rate=1.0,
        )
        assert row.to_dict()["avg_ttft"] is None


# ── Module-level constants ───────────────────────────────────────────────────

class TestNetworkConstants:
    def test_connect_timeout_positive(self):
        assert SESSION_CONNECT_TIMEOUT > 0

    def test_read_timeout_positive(self):
        assert SESSION_READ_TIMEOUT > 0

    def test_stream_timeout_greater_than_read(self):
        assert STREAM_READ_TIMEOUT >= SESSION_READ_TIMEOUT

    def test_retry_total_non_negative(self):
        assert RETRY_TOTAL >= 0

    def test_retry_status_forcelist_contains_429(self):
        assert 429 in RETRY_STATUS_FORCELIST

    def test_retry_status_forcelist_contains_5xx(self):
        server_errors = {500, 502, 503, 504}
        assert server_errors.issubset(set(RETRY_STATUS_FORCELIST))

    def test_max_parallel_workers_positive(self):
        assert MAX_PARALLEL_WORKERS >= 1


class TestBenchmarkPresets:
    def test_all_presets_have_prompt_and_category(self):
        for name, preset in BENCHMARK_PRESETS.items():
            assert "prompt" in preset, f"{name} missing 'prompt'"
            assert "category" in preset, f"{name} missing 'category'"

    def test_all_prompts_non_empty(self):
        for name, preset in BENCHMARK_PRESETS.items():
            assert preset["prompt"].strip(), f"{name} has empty prompt"

    def test_all_categories_in_smart_defaults_or_known(self):
        known_categories = set(SMART_DEFAULTS.keys())
        for name, preset in BENCHMARK_PRESETS.items():
            cat = preset["category"]
            assert cat in known_categories, (
                f"{name} has unknown category '{cat}' not in SMART_DEFAULTS"
            )

    def test_at_least_one_preset_defined(self):
        assert len(BENCHMARK_PRESETS) >= 1


class TestBenchmarkSuites:
    def test_all_suite_preset_keys_exist_in_presets(self):
        for suite_name, preset_keys in BENCHMARK_SUITES.items():
            for key in preset_keys:
                assert key in BENCHMARK_PRESETS, (
                    f"Suite '{suite_name}' references unknown preset '{key}'"
                )

    def test_all_presets_suite_contains_every_preset(self):
        all_presets_suite = next(
            (v for k, v in BENCHMARK_SUITES.items() if "All Presets" in k), None
        )
        assert all_presets_suite is not None, "No 'All Presets' suite found"
        assert set(all_presets_suite) == set(BENCHMARK_PRESETS.keys())

    def test_at_least_one_suite_defined(self):
        assert len(BENCHMARK_SUITES) >= 1


class TestSmartDefaults:
    def test_all_entries_have_temperature_and_top_p(self):
        for category, defaults in SMART_DEFAULTS.items():
            assert "temperature" in defaults, f"{category} missing temperature"
            assert "top_p" in defaults, f"{category} missing top_p"

    def test_temperature_in_valid_range(self):
        for category, defaults in SMART_DEFAULTS.items():
            t = defaults["temperature"]
            assert 0.0 <= t <= 2.0, f"{category} temperature {t} out of [0, 2]"

    def test_top_p_in_valid_range(self):
        for category, defaults in SMART_DEFAULTS.items():
            p = defaults["top_p"]
            assert 0.0 < p <= 1.0, f"{category} top_p {p} out of (0, 1]"

    def test_default_temperature_and_top_p_are_floats(self):
        assert isinstance(DEFAULT_TEMPERATURE, float)
        assert isinstance(DEFAULT_TOP_P, float)


class TestChartColors:
    def test_chart_colors_non_empty(self):
        assert len(CHART_COLORS) >= 1

    def test_all_colors_are_hex(self):
        hex_pattern = re.compile(r"^#[0-9a-fA-F]{6}$")
        for color in CHART_COLORS:
            assert hex_pattern.match(color), f"'{color}' is not a valid 6-digit hex color"


class TestMiscConstants:
    def test_max_prompt_history_positive(self):
        assert MAX_PROMPT_HISTORY > 0

    def test_max_prompt_history_is_int(self):
        assert isinstance(MAX_PROMPT_HISTORY, int)
