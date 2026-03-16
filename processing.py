"""
processing.py — Data Processing Module
───────────────────────────────────────
Isolated statistical aggregation and data transformation logic.

Responsibilities:
  • Aggregate raw BenchmarkResult lists into per-model statistics
  • Build typed LeaderboardRow objects with mean, stdev, CV%
  • Construct PyArrow-backed Pandas DataFrames for the UI leaderboard
  • Provide token estimation utility
  • Expose helper functions for preset/suite resolution

This module imports ONLY from `config` (internal), `statistics`, and `pandas`.
It never imports Gradio, Plotly, requests, or threading.

Pandas Configuration:
  • Copy-on-Write enabled (default in Pandas 3.0+)
  • PyArrow dtype backend for columnar memory layout and cache-local access
"""

from __future__ import annotations

import statistics as stats
from typing import Optional

import pandas as pd

from config import (
    BENCHMARK_PRESETS,
    BENCHMARK_SUITES,
    SMART_DEFAULTS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    MAX_PROMPT_HISTORY,
    PROMPT_PREVIEW_LEN,
    BenchmarkResult,
    LeaderboardRow,
)


# ── Pandas Engine Configuration ─────────────────────────────────────────────
# Copy-on-Write is default in Pandas 3.0+; explicit for clarity on 2.x.
pd.options.mode.copy_on_write = True


# ── Type Aliases ────────────────────────────────────────────────────────────

# Intermediate per-model accumulator used only inside this module.
_ModelAccumulator = dict[str, dict]


# ── Statistical Aggregation ─────────────────────────────────────────────────

def aggregate_stats(all_results: list[BenchmarkResult]) -> _ModelAccumulator:
    """
    Accumulate raw benchmark results into per-model statistical buckets.

    Returns:
        Dict keyed by model_id → {
            "name": str,
            "latencies": list[float],
            "ttfts": list[float],
            "tps_vals": list[float],
            "comp_tokens": list[int],
            "errors": int,
            "responses": list[str],
            "total_runs": int,
        }
    """
    model_stats: _ModelAccumulator = {}

    for result in all_results:
        mid = result.model_id

        if mid not in model_stats:
            model_stats[mid] = {
                "name": result.model_name,
                "latencies": [],
                "ttfts": [],
                "tps_vals": [],
                "comp_tokens": [],
                "errors": 0,
                "responses": [],
                "total_runs": 0,
            }

        bucket = model_stats[mid]
        bucket["total_runs"] += 1

        if result.is_error:
            bucket["errors"] += 1
        else:
            bucket["latencies"].append(result.latency_sec)
            if result.ttft_sec is not None:
                bucket["ttfts"].append(result.ttft_sec)
            bucket["tps_vals"].append(result.tokens_per_sec)
            bucket["comp_tokens"].append(result.completion_tokens)
            bucket["responses"].append(result.response)

    return model_stats


def build_leaderboard_rows(model_stats: _ModelAccumulator) -> list[LeaderboardRow]:
    """
    Transform accumulated stats into sorted LeaderboardRow objects.

    Sorting: descending by avg_tps (highest throughput first).

    Statistical methods:
      • Mean via statistics.mean (numerically stable)
      • Stdev via statistics.stdev (requires n ≥ 2)
      • CV% = (σ / μ) × 100 — coefficient of variation for normalized consistency
    """
    rows: list[LeaderboardRow] = []

    for mid, bucket in model_stats.items():
        tps_vals = bucket["tps_vals"]
        latencies = bucket["latencies"]
        ttfts = bucket["ttfts"]
        comp_tokens = bucket["comp_tokens"]
        total_runs = bucket["total_runs"]
        errors = bucket["errors"]

        avg_tps = stats.mean(tps_vals) if tps_vals else 0.0
        std_tps = stats.stdev(tps_vals) if len(tps_vals) >= 2 else 0.0
        cv_tps = (std_tps / avg_tps * 100.0) if avg_tps > 0 else 0.0

        avg_lat = stats.mean(latencies) if latencies else 999.0
        std_lat = stats.stdev(latencies) if len(latencies) >= 2 else 0.0

        avg_ttft: Optional[float] = (
            round(stats.mean(ttfts), 3) if ttfts else None
        )

        avg_tokens = round(stats.mean(comp_tokens)) if comp_tokens else 0

        error_rate = round(errors / total_runs * 100, 1) if total_runs else 0.0

        rows.append(
            LeaderboardRow(
                model=bucket["name"],
                model_id=mid,
                avg_lat=round(avg_lat, 2),
                std_lat=round(std_lat, 2),
                avg_ttft=avg_ttft,
                avg_tps=round(avg_tps, 1),
                std_tps=round(std_tps, 1),
                cv_tps=round(cv_tps, 1),
                avg_tokens=avg_tokens,
                errors=errors,
                runs=total_runs,
                error_rate=error_rate,
            )
        )

    rows.sort(key=lambda r: r.avg_tps, reverse=True)
    return rows


# ── DataFrame Construction (PyArrow-backed) ─────────────────────────────────

def build_leaderboard_dataframe(rows: list[LeaderboardRow]) -> pd.DataFrame:
    """
    Construct a display-ready DataFrame from LeaderboardRow objects.

    Uses PyArrow-backed nullable types for:
      • Cache-local columnar memory layout
      • Native nullable float (TTFT can be None → pd.NA, not NaN)
      • Reduced memory footprint vs object-dtype fallbacks

    The DataFrame is shaped for direct consumption by gr.Dataframe.
    """
    if not rows:
        return pd.DataFrame(
            columns=[
                "#", "Model", "Latency (s)", "σ Lat", "TTFT (s)",
                "tok/s", "σ tok/s", "CV%", "Tokens", "Errors", "Runs",
            ]
        )

    records: list[dict] = []
    for rank, row in enumerate(rows, 1):
        medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, str(rank))
        records.append({
            "#": medal,
            "Model": row.model,
            "Latency (s)": row.avg_lat,
            "σ Lat": row.std_lat,
            "TTFT (s)": row.avg_ttft,
            "tok/s": row.avg_tps,
            "σ tok/s": row.std_tps,
            "CV%": row.cv_tps,
            "Tokens": row.avg_tokens,
            "Errors": row.errors,
            "Runs": row.runs,
        })

    df = pd.DataFrame(records)

    # ── PyArrow type coercion ───────────────────────────────────────────
    # Explicit conversion ensures columnar layout and nullable semantics.
    arrow_dtypes: dict[str, str] = {
        "#": "string[pyarrow]",
        "Model": "string[pyarrow]",
        "Latency (s)": "float64[pyarrow]",
        "σ Lat": "float64[pyarrow]",
        "TTFT (s)": "float64[pyarrow]",
        "tok/s": "float64[pyarrow]",
        "σ tok/s": "float64[pyarrow]",
        "CV%": "float64[pyarrow]",
        "Tokens": "int64[pyarrow]",
        "Errors": "int64[pyarrow]",
        "Runs": "int64[pyarrow]",
    }

    for col, dtype in arrow_dtypes.items():
        if col in df.columns:
            df[col] = df[col].astype(dtype)

    return df


# ── Side-by-Side Response Markdown ──────────────────────────────────────────

def build_sidebyside_markdown(
    model_stats: _ModelAccumulator,
    max_pairs: int = 4,
    max_response_len: int = 1200,
) -> str:
    """
    Build HTML-table Markdown for side-by-side model response comparison.

    Pairs adjacent models that have at least one non-empty response.
    Falls back to single-model display if only one model responded.
    """
    ids_with_responses = [
        mid for mid, bucket in model_stats.items()
        if bucket["responses"]
    ]

    if not ids_with_responses:
        return "*No model responses to compare.*"

    parts: list[str] = []

    if len(ids_with_responses) >= 2:
        pairs_shown = 0
        for idx in range(len(ids_with_responses) - 1):
            if pairs_shown >= max_pairs:
                break

            mid_a = ids_with_responses[idx]
            mid_b = ids_with_responses[idx + 1]
            bucket_a = model_stats[mid_a]
            bucket_b = model_stats[mid_b]

            resp_a = bucket_a["responses"][0][:max_response_len]
            resp_b = bucket_b["responses"][0][:max_response_len]

            parts.append(
                f"### {bucket_a['name']} vs {bucket_b['name']}\n\n"
                f"<table><tr>"
                f"<td style='width:50%;vertical-align:top;padding:8px;'>"
                f"<strong>{bucket_a['name']}</strong><br>"
                f"<pre style='white-space:pre-wrap;font-size:0.85em;'>"
                f"{resp_a}</pre></td>"
                f"<td style='width:50%;vertical-align:top;padding:8px;'>"
                f"<strong>{bucket_b['name']}</strong><br>"
                f"<pre style='white-space:pre-wrap;font-size:0.85em;'>"
                f"{resp_b}</pre></td>"
                f"</tr></table>\n\n---\n\n"
            )
            pairs_shown += 1
    else:
        for mid in ids_with_responses:
            bucket = model_stats[mid]
            resp = bucket["responses"][0][:2000]
            parts.append(
                f"### {bucket['name']}\n\n```\n{resp}\n```\n\n---\n\n"
            )

    return "".join(parts)


# ── Token Estimation ────────────────────────────────────────────────────────

def estimate_tokens(text: str) -> str:
    """
    Approximate token count for display in the UI.

    Heuristic: ~1 token per 4 characters for English text.
    Returns a formatted string like "~142 tokens".
    """
    if not text:
        return "~0 tokens"
    estimated = max(1, len(text) * 10 // 40)
    return f"~{estimated} tokens"


# ── Preset / Suite Resolution ───────────────────────────────────────────────

def resolve_preset_prompt(preset_name: str) -> str:
    """Return the prompt text for a named preset, or empty string."""
    preset = BENCHMARK_PRESETS.get(preset_name)
    return preset["prompt"] if preset else ""


def resolve_smart_defaults(preset_name: str) -> tuple[float, float]:
    """
    Return (temperature, top_p) for a preset's category.

    Falls back to global defaults if the preset or category is unknown.
    """
    preset = BENCHMARK_PRESETS.get(preset_name)
    if preset:
        category = preset.get("category", "")
        defaults = SMART_DEFAULTS.get(category, {})
        return (
            defaults.get("temperature", DEFAULT_TEMPERATURE),
            defaults.get("top_p", DEFAULT_TOP_P),
        )
    return DEFAULT_TEMPERATURE, DEFAULT_TOP_P


def resolve_suite_prompts(suite_name: Optional[str]) -> list[tuple[str, str]]:
    """
    Expand a suite name into a list of (prompt_text, preset_label) tuples.

    Returns:
        Non-empty list if suite_name is valid.
        Empty list if suite_name is None, empty, or unrecognized.
    """
    if not suite_name or suite_name not in BENCHMARK_SUITES:
        return []

    prompts: list[tuple[str, str]] = []
    for preset_key in BENCHMARK_SUITES[suite_name]:
        preset = BENCHMARK_PRESETS.get(preset_key)
        if preset:
            prompts.append((preset["prompt"], preset_key))

    return prompts


# ── Prompt History Management ───────────────────────────────────────────────

def update_prompt_history(
    history: list[str],
    new_prompts: list[str],
) -> list[str]:
    """
    Prepend new prompts to session history, deduplicating and capping length.

    Returns a new list (no mutation of the input).
    """
    updated = list(history)
    for prompt in new_prompts:
        if prompt not in updated:
            updated.insert(0, prompt)
    return updated[:MAX_PROMPT_HISTORY]


def history_to_choices(history: list[str]) -> list[tuple[str, str]]:
    """
    Convert prompt history into (display_label, value) tuples
    suitable for a Gradio Dropdown.
    """
    choices: list[tuple[str, str]] = []
    for prompt in history:
        if len(prompt) > PROMPT_PREVIEW_LEN:
            label = f"{prompt[:PROMPT_PREVIEW_LEN]}..."
        else:
            label = prompt
        choices.append((label, prompt))
    return choices


# ── Blind Mode Name Resolution ──────────────────────────────────────────────

def apply_blind_labels(
    selected_model_ids: list[str],
    model_id: str,
) -> str:
    """
    Return an anonymized label for blind mode.

    Maps model_id to its position index in the selection list.
    """
    try:
        idx = selected_model_ids.index(model_id)
    except ValueError:
        idx = 0
    return f"Model #{idx + 1}"


def reveal_blind_results(
    all_results: list[BenchmarkResult],
    model_lookup: dict[str, str],
) -> None:
    """
    Mutate BenchmarkResult.model_name in-place, replacing blind labels
    with real model names after the benchmark completes.

    Args:
        all_results: List of results with anonymized names.
        model_lookup: Dict of model_id → real display name.
    """
    for result in all_results:
        real_name = model_lookup.get(result.model_id)
        if real_name:
            result.model_name = real_name
