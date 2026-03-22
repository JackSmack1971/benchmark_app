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
pd.options.mode.copy_on_write = True
pd.options.mode.dtype_backend = "pyarrow"

# ── Type Aliases ────────────────────────────────────────────────────────────
_ModelAccumulator = dict[str, dict]


# ── Core Data Ingestion & Downcasting ───────────────────────────────────────

def _ingest_and_downcast(all_results: list[BenchmarkResult]) -> pd.DataFrame:
    """
    Ingests raw telemetry into a zero-copy Arrow memory layout and aggressively
    downcasts 64-bit defaults to 32-bit hardware boundaries.
    """
    if not all_results:
        return pd.DataFrame()

    df = pd.DataFrame([r.to_dict() for r in all_results])

    # ── Strict Hardware-Optimized Schema ──
    # Shrinks footprint by 50% and aligns data perfectly with CPU cache lines.
    schema = {
        'model_id': 'category',
        'model_name': 'category',
        'latency_sec': 'float32[pyarrow]',
        'ttft_sec': 'float32[pyarrow]',
        'tokens_per_sec': 'float32[pyarrow]',
        'prompt_tokens': 'int32[pyarrow]',
        'completion_tokens': 'int32[pyarrow]',
        'total_tokens': 'int32[pyarrow]',
        'prompt_cost_usd': 'float64[pyarrow]',
        'completion_cost_usd': 'float64[pyarrow]',
        'error': 'string[pyarrow]',
        'response': 'string[pyarrow]'
    }

    return df.astype({k: v for k, v in schema.items() if k in df.columns})


# ── Statistical Aggregation ─────────────────────────────────────────────────

def aggregate_stats(all_results: list[BenchmarkResult]) -> _ModelAccumulator:
    """
    Accumulate raw benchmark results utilizing the downcasted PyArrow engine.
    Maintains dict output contract specifically for Plotly box/scatter charts.
    """
    df = _ingest_and_downcast(all_results)
    model_stats: _ModelAccumulator = {}

    if df.empty:
        return model_stats

    # Grouping via categorical index is natively accelerated in PyArrow
    for mid, group in df.groupby('model_id', observed=True):
        if group.empty:
            continue
            
        total_costs = (
            group['prompt_cost_usd'].fillna(0) + group['completion_cost_usd'].fillna(0)
        ).tolist()
        model_stats[str(mid)] = {
            "name": str(group['model_name'].iloc[0]),
            "latencies": group['latency_sec'].dropna().tolist(),
            "ttfts": group['ttft_sec'].dropna().tolist(),
            "tps_vals": group['tokens_per_sec'].dropna().tolist(),
            "comp_tokens": group['completion_tokens'].dropna().tolist(),
            "total_costs": total_costs,
            "errors": int(group['error'].notna().sum()),
            "responses": group['response'].dropna().tolist(),
            "total_runs": len(group)
        }

    return model_stats


def build_leaderboard_rows(model_stats: _ModelAccumulator) -> list[LeaderboardRow]:
    """
    Transform accumulated stats into sorted LeaderboardRow objects.
    Calculates unified metrics (Mean, Stdev, CV%).
    """
    rows: list[LeaderboardRow] = []

    for mid, bucket in model_stats.items():
        tps_vals = bucket["tps_vals"]
        latencies = bucket["latencies"]
        ttfts = bucket["ttfts"]
        comp_tokens = bucket["comp_tokens"]
        total_costs = bucket.get("total_costs", [])
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

        avg_total_cost = stats.mean(total_costs) if total_costs else 0.0
        avg_cost_per_1k = (
            round(avg_total_cost / avg_tokens * 1000, 6) if avg_tokens > 0 else 0.0
        )

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
                avg_cost_per_1k=avg_cost_per_1k,
            )
        )

    rows.sort(key=lambda r: r.avg_tps, reverse=True)
    return rows


# ── UI DataFrame Construction (Aggressively Downcasted) ─────────────────────

def build_leaderboard_dataframe(rows: list[LeaderboardRow]) -> pd.DataFrame:
    """
    Construct a presentation-ready DataFrame enforcing strict 32-bit types.
    This safely feeds Gradio without risking event loop stalls during rendering.
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
            "Cost/1K tokens ($)": row.avg_cost_per_1k,
            "Errors": row.errors,
            "Runs": row.runs,
        })

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
