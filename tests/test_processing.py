"""
Tests for processing.py — achieving 100% line coverage gate.
"""
import pytest
import pandas as pd
from collections import defaultdict
import statistics as stats

from config import (
    BenchmarkResult,
    LeaderboardRow,
    ModelInfo,
    BENCHMARK_PRESETS,
    BENCHMARK_SUITES,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P
)

from aggregation import (
    _ingest_and_downcast,
    aggregate_stats,
    build_leaderboard_rows,
    compute_radar_scores
)
from processing import (
    build_leaderboard_dataframe,
    build_category_breakdown,
    build_sidebyside_markdown,
    estimate_tokens,
    resolve_preset_prompt,
    resolve_smart_defaults,
    resolve_suite_prompts,
    update_prompt_history,
    history_to_choices,
    apply_blind_labels,
    reveal_blind_results
)

def test_ingest_and_downcast_empty():
    df = _ingest_and_downcast([])
    assert df.empty

def test_ingest_and_downcast_valid(sample_result):
    df = _ingest_and_downcast([sample_result])
    assert not df.empty
    assert str(df.dtypes['latency_sec']) in ('float32[pyarrow]', 'float[pyarrow]')
    assert str(df.dtypes['model_id']) == 'category'

def test_aggregate_stats_empty():
    assert aggregate_stats([]) == {}

def test_aggregate_stats_valid(sample_result, error_result):
    stats_dict = aggregate_stats([sample_result, error_result, sample_result])
    assert len(stats_dict) == 1
    bucket = stats_dict[sample_result.model_id]
    assert bucket["name"] == sample_result.model_name
    assert bucket["total_runs"] == 3
    assert bucket["errors"] == 1
    assert len(bucket["latencies"]) == 3  # The error_result has 0.0 but let's check config, wait error_result has error string. 
    # Actually _ingest_and_downcast does not drop valid 0.0s, but wait, error_result has latency_sec = 0.0. aggregate drops NA?
    # It drops NA, but 0.0 is not NA. Let's see if 0.0 ends up in latencies. Yes it does.

def test_build_leaderboard_rows_empty():
    assert build_leaderboard_rows({}) == []

def test_build_leaderboard_rows_valid(sample_result, error_result):
    # Single model stats
    rows = build_leaderboard_rows([sample_result, sample_result, error_result])
    assert len(rows) == 1
    row = rows[0]
    assert row.runs == 3
    assert row.errors == 1
    assert row.error_rate == pytest.approx(33.3, 0.1)

def test_compute_radar_scores_empty():
    assert compute_radar_scores([]) == {}

def test_compute_radar_scores_valid(sample_leaderboard_row):
    # One good row, one bad row
    row_bad = LeaderboardRow(
        model="Bad Model",
        model_id="bad",
        avg_lat=10.0,
        std_lat=1.0,
        avg_ttft=5.0,
        avg_tps=0.5,
        std_tps=0.1,
        cv_tps=20.0,
        avg_tokens=10,
        errors=1,
        runs=1,
        error_rate=100.0
    )
    scores = compute_radar_scores([sample_leaderboard_row, row_bad])
    assert scores[sample_leaderboard_row.model_id] > scores[row_bad.model_id]

def test_compute_radar_scores_zeros():
    row_zero = LeaderboardRow(
        model="Zero Model", model_id="zero",
        avg_lat=0, std_lat=0, avg_ttft=0, avg_tps=0, std_tps=0, cv_tps=0, avg_tokens=0, errors=0, runs=1, error_rate=0
    )
    scores = compute_radar_scores([row_zero])
    assert isinstance(scores["zero"], float)

def test_build_leaderboard_dataframe_empty():
    df = build_leaderboard_dataframe([])
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["#", "Model", "Latency (s)", "σ Lat", "TTFT (s)", "tok/s", "σ tok/s", "CV%", "Tokens", "Cost/1K tokens ($)", "Errors", "Runs"]

def test_build_leaderboard_dataframe_valid(sample_leaderboard_row, sample_model_info):
    # Test mapped context length
    row1 = sample_leaderboard_row
    row2 = LeaderboardRow(
        model="No Info", model_id="no/info",
        avg_lat=1, std_lat=0, avg_ttft=1, avg_tps=1, std_tps=0, cv_tps=0, avg_tokens=100, errors=0, runs=1, error_rate=0
    )
    df = build_leaderboard_dataframe([row1, row2], model_info_map={row1.model_id: sample_model_info})
    assert len(df) == 2
    ctx_series = df["Ctx Used"].tolist()
    assert ctx_series[0] != "—"  # Uses the mapping
    assert ctx_series[1] == "—"  # No mapping found

    # Type coercion check
    assert str(df.dtypes['#']) in ('string[pyarrow]', 'string')
    assert str(df.dtypes['Tokens']) in ('int32[pyarrow]', 'int32', 'int64')

def test_build_category_breakdown_empty():
    assert build_category_breakdown([]).empty

def test_build_category_breakdown_no_suitelabel(sample_result):
    sample_result.suite_label = ""
    df = build_category_breakdown([sample_result])
    assert df.empty

def test_build_category_breakdown_valid(sample_result):
    sample_result.suite_label = "Test Suite"
    df = build_category_breakdown([sample_result])
    assert not df.empty
    assert list(df.columns) == ["model_name", "suite_label", "avg_tps", "avg_lat"]

def test_build_sidebyside_markdown():
    stats_empty = {}
    assert "No model responses" in build_sidebyside_markdown(stats_empty)

    stats_one = {"mid1": {"name": "M1", "responses": ["Hello 1"]}}
    md = build_sidebyside_markdown(stats_one)
    assert "### M1" in md
    
    stats_two = {
        "mid1": {"name": "M1", "responses": ["Hello 1"]},
        "mid2": {"name": "M2", "responses": ["Hello 2"]}
    }
    md = build_sidebyside_markdown(stats_two)
    assert "M1 vs M2" in md
    
    # 3 items to test multiple pairs
    stats_three = {
        "m1": {"name": "M1", "responses": ["R1"]},
        "m2": {"name": "M2", "responses": ["R2"]},
        "m3": {"name": "M3", "responses": ["R3"]}
    }
    md = build_sidebyside_markdown(stats_three, max_pairs=1)
    # Should only show 1 pair
    assert "M1 vs M2" in md
    assert "M2 vs M3" not in md

def test_estimate_tokens():
    assert estimate_tokens("") == "~0 tokens"
    assert estimate_tokens("A B C D E") == "~2 tokens"

def test_resolve_preset_prompt():
    assert resolve_preset_prompt("⚡ Quick Reasoning") != ""
    assert resolve_preset_prompt("Nonexistent") == ""

def test_resolve_smart_defaults():
    t, p = resolve_smart_defaults("⚡ Quick Reasoning")
    assert t == 0.3
    assert p == 0.9
    # fallback
    t, p = resolve_smart_defaults("Nonexistent")
    assert t == DEFAULT_TEMPERATURE
    assert p == DEFAULT_TOP_P

def test_resolve_suite_prompts():
    assert resolve_suite_prompts("") == []
    assert resolve_suite_prompts("Invalid") == []
    prompts = resolve_suite_prompts("🧪 Full Reasoning Suite")
    assert len(prompts) > 0

def test_update_prompt_history():
    hist = ["A", "B", "C"]
    # Add existing
    assert update_prompt_history(hist, ["A"]) == ["A", "B", "C"]
    # Add new
    assert update_prompt_history(hist, ["D"])[0] == "D"

def test_history_to_choices():
    hist = ["short", "a" * 100]
    choices = history_to_choices(hist)
    assert len(choices) == 2
    assert choices[0] == ("short", "short")
    assert "..." in choices[1][0]

def test_apply_blind_labels():
    assert apply_blind_labels(["a", "b"], "a") == "Model #1"
    assert apply_blind_labels(["a", "b"], "b") == "Model #2"
    assert apply_blind_labels(["a", "b"], "c") == "Model #1"

def test_reveal_blind_results(sample_result):
    sample_result.model_name = "Model #1"
    reveal_blind_results([sample_result], {sample_result.model_id: "Real Name"})
    assert sample_result.model_name == "Real Name"

    # Not in lookup
    sample_result.model_name = "Model #1"
    reveal_blind_results([sample_result], {"other": "Real"})
    assert sample_result.model_name == "Model #1"
