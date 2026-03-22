"""
processing.py — Data Processing Module (Hardware-Optimized Edition)
───────────────────────────────────────────────────────────────────
Isolated statistical aggregation utilizing zero-copy Apache Arrow memory.

Architectural Enhancements:
  • Aggressive 32-bit downcasting (int32/float32) for L1/L2 cache optimization.
  • Categorical type coercion for low-cardinality string identifiers.
  • Vectorized method chaining to eliminate scalar execution bottlenecks.
"""

from __future__ import annotations

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
    ModelInfo,
)

# ── Pandas Engine Configuration ─────────────────────────────────────────────
pd.options.mode.copy_on_write = True

# We need _ingest_and_downcast and compute_radar_scores for internal references
# but mostly we'll import what we need or provide inline implementations.
# Actually, we need to import _ingest_and_downcast from aggregation to process lists for other functions like category breakdown.
from aggregation import _ingest_and_downcast, _ModelAccumulator





# ── UI DataFrame Construction (Aggressively Downcasted) ─────────────────────

def build_leaderboard_dataframe(
    rows: list[LeaderboardRow],
    model_info_map: Optional[dict[str, ModelInfo]] = None,
) -> pd.DataFrame:
    """
    Construct a presentation-ready DataFrame enforcing strict 32-bit types.
    This safely feeds Gradio without risking event loop stalls during rendering.

    Args:
        rows: Sorted leaderboard rows.
        model_info_map: Optional {model_id: ModelInfo} for context window column.
    """
    if not rows:
        return pd.DataFrame(
            columns=[
                "#", "Model", "Latency (s)", "σ Lat", "TTFT (s)",
                "tok/s", "σ tok/s", "CV%", "Tokens", "Cost/1K tokens ($)", "Errors", "Runs",
            ]
        )

    records: list[dict] = []
    for rank, row in enumerate(rows, 1):
        medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, str(rank))
        record: dict = {
            "#": medal,
            "Model": row.model,
            "Latency (s)": row.avg_lat,
            "σ Lat": row.std_lat,
            "TTFT (s)": row.avg_ttft,
            "tok/s": row.avg_tps,
            "σ tok/s": row.std_tps,
            "CV%": row.cv_tps,
            "Tokens": row.avg_tokens,
            "Cost/1K tokens ($)": row.avg_cost_per_1k,
            "Errors": row.errors,
            "Runs": row.runs,
        }
        if model_info_map:
            info = model_info_map.get(row.model_id)
            if info and info.context_length > 0:
                pct = round(row.avg_tokens / info.context_length * 100, 1)
                ctx_k = f"{info.context_length // 1000}K" if info.context_length >= 1000 else str(info.context_length)
                record["Ctx Used"] = f"{pct}% of {ctx_k}"
            else:
                record["Ctx Used"] = "—"
        records.append(record)

    df = pd.DataFrame(records)

    # ── Final Layer Type Coercion ──
    arrow_dtypes: dict[str, str] = {
        "#": "string[pyarrow]",
        "Model": "category",                       # Memory-mapped string references
        "Latency (s)": "float32[pyarrow]",         # 32-bit float boundaries
        "σ Lat": "float32[pyarrow]",
        "TTFT (s)": "float32[pyarrow]",
        "tok/s": "float32[pyarrow]",
        "σ tok/s": "float32[pyarrow]",
        "CV%": "float32[pyarrow]",
        "Tokens": "int32[pyarrow]",                # 32-bit integer boundaries
        "Cost/1K tokens ($)": "float64[pyarrow]",  # 64-bit for sub-cent precision
        "Errors": "int32[pyarrow]",
        "Runs": "int32[pyarrow]",
        "Ctx Used": "string[pyarrow]",
    }

    for col, dtype in arrow_dtypes.items():
        if col in df.columns:
            df[col] = df[col].astype(dtype)

    return df


# ── Per-Category Breakdown ──────────────────────────────────────────────────

def build_category_breakdown(all_results: list[BenchmarkResult]) -> pd.DataFrame:
    """
    Group benchmark results by (model_name, suite_label) and compute mean tok/s
    and mean latency per cell. Only includes results with a non-empty suite_label.

    Returns:
        DataFrame with columns: model_name, suite_label, avg_tps, avg_lat.
        Empty DataFrame if no suite data is present.
    """
    df = _ingest_and_downcast(all_results)
    if df.empty:
        return pd.DataFrame()

    # Filter to rows that were part of a named suite/preset
    if "suite_label" not in df.columns:
        return pd.DataFrame()

    suite_df = df[df["suite_label"].astype(str).str.strip() != ""]
    if suite_df.empty:
        return pd.DataFrame()

    grouped = (
        suite_df.groupby(["model_name", "suite_label"], observed=True)
        .agg(
            avg_tps=("tokens_per_sec", "mean"),
            avg_lat=("latency_sec", "mean"),
        )
        .reset_index()
    )

    grouped["avg_tps"] = grouped["avg_tps"].round(1)
    grouped["avg_lat"] = grouped["avg_lat"].round(2)
    grouped["model_name"] = grouped["model_name"].astype(str)
    grouped["suite_label"] = grouped["suite_label"].astype(str)

    return grouped


# ── Side-by-Side Response Markdown ──────────────────────────────────────────

def build_sidebyside_markdown(
    model_stats: _ModelAccumulator,
    max_pairs: int = 4,
    max_response_len: int = 1200,
) -> str:
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


# ── Utility Functions ───────────────────────────────────────────────────────

def estimate_tokens(text: str) -> str:
    if not text:
        return "~0 tokens"
    estimated = max(1, len(text) * 10 // 40)
    return f"~{estimated} tokens"


def resolve_preset_prompt(preset_name: str) -> str:
    preset = BENCHMARK_PRESETS.get(preset_name)
    return preset["prompt"] if preset else ""


def resolve_smart_defaults(preset_name: str) -> tuple[float, float]:
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
    if not suite_name or suite_name not in BENCHMARK_SUITES:
        return []

    prompts: list[tuple[str, str]] = []
    for preset_key in BENCHMARK_SUITES[suite_name]:
        preset = BENCHMARK_PRESETS.get(preset_key)
        if preset:
            prompts.append((preset["prompt"], preset_key))

    return prompts


def update_prompt_history(history: list[str], new_prompts: list[str]) -> list[str]:
    updated = list(history)
    for prompt in new_prompts:
        if prompt not in updated:
            updated.insert(0, prompt)
    return updated[:MAX_PROMPT_HISTORY]


def history_to_choices(history: list[str]) -> list[tuple[str, str]]:
    choices: list[tuple[str, str]] = []
    for prompt in history:
        label = f"{prompt[:PROMPT_PREVIEW_LEN]}..." if len(prompt) > PROMPT_PREVIEW_LEN else prompt
        choices.append((label, prompt))
    return choices


def apply_blind_labels(selected_model_ids: list[str], model_id: str) -> str:
    try:
        idx = selected_model_ids.index(model_id)
    except ValueError:
        idx = 0
    return f"Model #{idx + 1}"


def reveal_blind_results(all_results: list[BenchmarkResult], model_lookup: dict[str, str]) -> None:
    for result in all_results:
        real_name = model_lookup.get(result.model_id)
        if real_name:
            result.model_name = real_name
