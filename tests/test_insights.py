"""
test_insights.py — Unit tests for insights.py

Covers:
- Empty input returns empty string
- Single-row input (no comparisons possible)
- Fastest throughput always reported (rows[0])
- Lowest latency reported only when different model from throughput leader
- Lowest latency suppressed when same model as throughput leader
- Best TTFT reported when present; suppressed when all TTFT are None
- Consistency section only appears with cv_tps > 0
- Output volume spread: ≥30% threshold, zero-token guard, single-row guard
- Error warnings: single and multiple error models
- Markdown structure: header present, sections joined with double newline
- Ties: two models with identical throughput (rows[0] wins by sort order)
"""

import pytest

from config import LeaderboardRow
from insights import generate_insights


# ── helpers ─────────────────────────────────────────────────────────────────

def make_row(
    model: str = "Model A",
    model_id: str = "org/model-a",
    avg_lat: float = 1.0,
    std_lat: float = 0.0,
    avg_ttft: float | None = 0.2,
    avg_tps: float = 50.0,
    std_tps: float = 0.0,
    cv_tps: float = 0.0,
    avg_tokens: int = 100,
    errors: int = 0,
    runs: int = 1,
    error_rate: float = 0.0,
) -> LeaderboardRow:
    return LeaderboardRow(
        model=model,
        model_id=model_id,
        avg_lat=avg_lat,
        std_lat=std_lat,
        avg_ttft=avg_ttft,
        avg_tps=avg_tps,
        std_tps=std_tps,
        cv_tps=cv_tps,
        avg_tokens=avg_tokens,
        errors=errors,
        runs=runs,
        error_rate=error_rate,
    )


# ── empty / single row ───────────────────────────────────────────────────────

class TestEmptyAndSingleRow:
    def test_empty_list_returns_empty_string(self):
        assert generate_insights([]) == ""

    def test_single_row_returns_non_empty_string(self):
        result = generate_insights([make_row()])
        assert result != ""

    def test_single_row_contains_header(self):
        result = generate_insights([make_row()])
        assert "## 🔎 Insight Summary" in result

    def test_single_row_contains_throughput(self):
        result = generate_insights([make_row(model="Solo", avg_tps=42.0)])
        assert "Solo" in result
        assert "42.0 tok/s" in result

    def test_single_row_no_latency_section(self):
        # Latency leader == throughput leader when only one row — suppressed
        result = generate_insights([make_row(model="Solo")])
        assert "Lowest latency" not in result

    def test_single_row_no_output_spread(self):
        result = generate_insights([make_row()])
        assert "Output spread" not in result


# ── fastest throughput ───────────────────────────────────────────────────────

class TestFastestThroughput:
    def test_fastest_throughput_is_first_row(self):
        rows = [
            make_row(model="Fast", avg_tps=100.0),
            make_row(model="Slow", avg_tps=20.0),
        ]
        result = generate_insights(rows)
        assert "Fast" in result
        assert "100.0 tok/s" in result

    def test_throughput_label_present(self):
        result = generate_insights([make_row(model="A", avg_tps=55.5)])
        assert "Fastest throughput" in result


# ── lowest latency ───────────────────────────────────────────────────────────

class TestLowestLatency:
    def test_latency_section_shown_when_different_model(self):
        rows = [
            make_row(model="HighTPS", avg_tps=100.0, avg_lat=2.0),
            make_row(model="LowLat",  avg_tps=50.0,  avg_lat=0.5),
        ]
        result = generate_insights(rows)
        assert "Lowest latency" in result
        assert "LowLat" in result

    def test_latency_value_shown(self):
        rows = [
            make_row(model="HighTPS", avg_tps=100.0, avg_lat=3.0),
            make_row(model="LowLat",  avg_tps=50.0,  avg_lat=0.4),
        ]
        result = generate_insights(rows)
        assert "0.4s" in result

    def test_latency_section_suppressed_when_same_model(self):
        # throughput leader also has the lowest latency
        rows = [
            make_row(model="Best",  avg_tps=100.0, avg_lat=0.5),
            make_row(model="Other", avg_tps=50.0,  avg_lat=2.0),
        ]
        result = generate_insights(rows)
        assert "Lowest latency" not in result

    def test_latency_section_suppressed_for_single_row(self):
        result = generate_insights([make_row(model="Only", avg_tps=10.0, avg_lat=1.0)])
        assert "Lowest latency" not in result


# ── best TTFT ────────────────────────────────────────────────────────────────

class TestBestTTFT:
    def test_ttft_section_shown_when_present(self):
        rows = [
            make_row(model="A", avg_tps=100.0, avg_ttft=0.5),
            make_row(model="B", avg_tps=50.0,  avg_ttft=0.1),
        ]
        result = generate_insights(rows)
        assert "Fastest first token" in result
        assert "B" in result
        assert "0.1s TTFT" in result

    def test_ttft_section_suppressed_when_all_none(self):
        rows = [
            make_row(model="A", avg_ttft=None),
            make_row(model="B", avg_ttft=None),
        ]
        result = generate_insights(rows)
        assert "Fastest first token" not in result

    def test_ttft_section_suppressed_for_single_row_with_none(self):
        result = generate_insights([make_row(avg_ttft=None)])
        assert "Fastest first token" not in result

    def test_ttft_picks_minimum_not_first_row(self):
        rows = [
            make_row(model="Slow TTFT", avg_tps=100.0, avg_ttft=1.0),
            make_row(model="Fast TTFT", avg_tps=50.0,  avg_ttft=0.05),
        ]
        result = generate_insights(rows)
        assert "Fast TTFT" in result
        assert "0.05s TTFT" in result

    def test_ttft_partial_none_uses_available_values(self):
        rows = [
            make_row(model="NoTTFT", avg_tps=100.0, avg_ttft=None),
            make_row(model="HasTTFT", avg_tps=50.0,  avg_ttft=0.3),
        ]
        result = generate_insights(rows)
        assert "Fastest first token" in result
        assert "HasTTFT" in result


# ── consistency ──────────────────────────────────────────────────────────────

class TestConsistency:
    def test_consistency_shown_when_cv_nonzero(self):
        rows = [
            make_row(model="Steady", avg_tps=100.0, cv_tps=2.0),
            make_row(model="Wobbly", avg_tps=80.0,  cv_tps=15.0),
        ]
        result = generate_insights(rows)
        assert "Most consistent" in result
        assert "Steady" in result
        assert "2.0% CV" in result

    def test_consistency_suppressed_when_all_cv_zero(self):
        rows = [
            make_row(model="A", cv_tps=0.0),
            make_row(model="B", cv_tps=0.0),
        ]
        result = generate_insights(rows)
        assert "Most consistent" not in result

    def test_consistency_picks_lowest_cv(self):
        rows = [
            make_row(model="High CV", avg_tps=100.0, cv_tps=20.0),
            make_row(model="Low CV",  avg_tps=90.0,  cv_tps=1.5),
            make_row(model="Mid CV",  avg_tps=80.0,  cv_tps=8.0),
        ]
        result = generate_insights(rows)
        assert "Low CV" in result

    def test_consistency_ignores_zero_cv_rows(self):
        rows = [
            make_row(model="SingleRun", avg_tps=100.0, cv_tps=0.0),
            make_row(model="MultiRun",  avg_tps=80.0,  cv_tps=5.0),
        ]
        result = generate_insights(rows)
        assert "MultiRun" in result
        # The zero-CV model must not appear as the consistency leader
        consistency_line = result.split("Most consistent")[1].split("\n")[0]
        assert "SingleRun" not in consistency_line


# ── output volume spread ─────────────────────────────────────────────────────

class TestOutputSpread:
    def test_spread_shown_when_over_30_percent(self):
        rows = [
            make_row(model="Verbose", avg_tps=100.0, avg_tokens=200),
            make_row(model="Terse",   avg_tps=80.0,  avg_tokens=100),
        ]
        result = generate_insights(rows)
        assert "Output spread" in result
        assert "Verbose" in result
        assert "Terse" in result

    def test_spread_suppressed_when_under_30_percent(self):
        rows = [
            make_row(model="A", avg_tps=100.0, avg_tokens=110),
            make_row(model="B", avg_tps=80.0,  avg_tokens=100),
        ]
        result = generate_insights(rows)
        assert "Output spread" not in result

    def test_spread_suppressed_when_least_verbose_has_zero_tokens(self):
        rows = [
            make_row(model="A", avg_tps=100.0, avg_tokens=500),
            make_row(model="B", avg_tps=80.0,  avg_tokens=0),
        ]
        result = generate_insights(rows)
        assert "Output spread" not in result

    def test_spread_suppressed_for_single_row(self):
        result = generate_insights([make_row(avg_tokens=999)])
        assert "Output spread" not in result

    def test_spread_at_exact_30_percent_boundary_suppressed(self):
        # most = least * 1.3 exactly → condition is strictly greater, so suppressed
        rows = [
            make_row(model="A", avg_tps=100.0, avg_tokens=130),
            make_row(model="B", avg_tps=80.0,  avg_tokens=100),
        ]
        result = generate_insights(rows)
        assert "Output spread" not in result

    def test_spread_shown_just_above_30_percent(self):
        rows = [
            make_row(model="A", avg_tps=100.0, avg_tokens=131),
            make_row(model="B", avg_tps=80.0,  avg_tokens=100),
        ]
        result = generate_insights(rows)
        assert "Output spread" in result

    def test_spread_token_counts_in_output(self):
        rows = [
            make_row(model="Wordy",  avg_tps=100.0, avg_tokens=300),
            make_row(model="Quiet",  avg_tps=80.0,  avg_tokens=50),
        ]
        result = generate_insights(rows)
        assert "300" in result
        assert "50" in result


# ── error warnings ───────────────────────────────────────────────────────────

class TestErrorWarnings:
    def test_error_section_shown_for_model_with_errors(self):
        rows = [
            make_row(model="Broken", errors=2, error_rate=0.5),
            make_row(model="Fine",   errors=0),
        ]
        result = generate_insights(rows)
        assert "Errors detected" in result
        assert "Broken (2)" in result

    def test_error_section_suppressed_when_no_errors(self):
        rows = [make_row(errors=0), make_row(model="B", errors=0)]
        result = generate_insights(rows)
        assert "Errors detected" not in result

    def test_multiple_error_models_listed(self):
        rows = [
            make_row(model="A", avg_tps=100.0, errors=1),
            make_row(model="B", avg_tps=80.0,  errors=3),
            make_row(model="C", avg_tps=60.0,  errors=0),
        ]
        result = generate_insights(rows)
        assert "A (1)" in result
        assert "B (3)" in result
        assert "C" not in result.split("Errors detected")[1]

    def test_error_section_for_single_errored_model(self):
        rows = [make_row(model="OnlyOne", errors=5)]
        result = generate_insights(rows)
        assert "OnlyOne (5)" in result


# ── markdown structure ───────────────────────────────────────────────────────

class TestMarkdownStructure:
    def test_header_is_first_section(self):
        result = generate_insights([make_row()])
        assert result.startswith("## 🔎 Insight Summary")

    def test_sections_separated_by_double_newline(self):
        rows = [
            make_row(model="Fast", avg_tps=100.0, avg_ttft=0.1),
            make_row(model="Slow", avg_tps=50.0,  avg_lat=0.5),
        ]
        result = generate_insights(rows)
        assert "\n\n" in result

    def test_return_type_is_str(self):
        assert isinstance(generate_insights([]), str)
        assert isinstance(generate_insights([make_row()]), str)


# ── tie-breaking ─────────────────────────────────────────────────────────────

class TestTies:
    def test_throughput_tie_uses_first_row(self):
        # Both have same avg_tps; rows[0] is treated as fastest
        rows = [
            make_row(model="First",  avg_tps=75.0),
            make_row(model="Second", avg_tps=75.0),
        ]
        result = generate_insights(rows)
        assert "First" in result
        assert "75.0 tok/s" in result

    def test_ttft_tie_returns_one_model(self):
        rows = [
            make_row(model="A", avg_tps=100.0, avg_ttft=0.2),
            make_row(model="B", avg_tps=80.0,  avg_ttft=0.2),
        ]
        result = generate_insights(rows)
        # One of them should appear as TTFT leader — no crash
        assert "Fastest first token" in result
        assert "0.2s TTFT" in result
