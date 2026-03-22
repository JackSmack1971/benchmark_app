"""
aggregation.py — Data Aggregation Module
────────────────────────────────────────
Isolated statistical aggregation utilizing zero-copy Apache Arrow memory.
"""

from __future__ import annotations

import pandas as pd
from typing import Optional

from config import BenchmarkResult, LeaderboardRow, ModelInfo

# ── Pandas Engine Configuration ─────────────────────────────────────────────
pd.options.mode.copy_on_write = True

_ModelAccumulator = dict[str, dict]

def _ingest_and_downcast(all_results: list[BenchmarkResult]) -> pd.DataFrame:
    """
    Ingests raw telemetry into a zero-copy Arrow memory layout and aggressively
    downcasts 64-bit defaults to 32-bit hardware boundaries.
    """
    if not all_results:
        return pd.DataFrame()

    df = pd.DataFrame([r.to_dict() for r in all_results])

    # ── Strict Hardware-Optimized Schema ──
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

def build_leaderboard_rows(all_results: list[BenchmarkResult]) -> list[LeaderboardRow]:
    """
    Transform raw results into sorted LeaderboardRow objects using PyArrow 
    vectorized groupby and column operations. Entirely bypasses scalar math.
    """
    df = _ingest_and_downcast(all_results)
    if df.empty:
        return []

    # Inject computable columns entirely in C++ Arrow structures
    df['total_cost'] = df['prompt_cost_usd'].fillna(0) + df['completion_cost_usd'].fillna(0)
    df['has_error'] = df['error'].notna().astype('int32[pyarrow]')

    # 1. Hardware-accelerated GroupBy
    grouped = df.groupby('model_id', observed=True).agg(
        model_name=('model_name', 'first'),
        runs=('model_id', 'size'),
        errors=('has_error', 'sum'),
        avg_tps=('tokens_per_sec', 'mean'),
        std_tps=('tokens_per_sec', 'std'),
        avg_lat=('latency_sec', 'mean'),
        std_lat=('latency_sec', 'std'),
        avg_ttft=('ttft_sec', 'mean'),
        avg_tokens=('completion_tokens', 'mean'),
        avg_total_cost=('total_cost', 'mean')
    )

    grouped = grouped[grouped['runs'] > 0].copy()

    # Fill NA where std deviation cannot be calculated (singleton runs)
    grouped['std_tps'] = grouped['std_tps'].fillna(0.0)
    grouped['std_lat'] = grouped['std_lat'].fillna(0.0)

    # 2. Method Chaining for derived scalar properties
    grouped = grouped.assign(
        error_rate=lambda x: (x['errors'] / x['runs'] * 100.0).round(1).fillna(0.0),
        cv_tps=lambda x: (x['std_tps'] / x['avg_tps'] * 100.0).fillna(0.0),
        avg_cost_per_1k=lambda x: (x['avg_total_cost'] / x['avg_tokens'] * 1000.0).fillna(0.0)
    )

    rows: list[LeaderboardRow] = []
    
    # 3. Output Translation
    for mid, g_row in grouped.iterrows():
        avg_lat_val = 999.0 if pd.isna(g_row['avg_lat']) else g_row['avg_lat']
        avg_ttft_val = None if pd.isna(g_row['avg_ttft']) else round(float(g_row['avg_ttft']), 3)

        row = LeaderboardRow(
            model=str(g_row['model_name']),
            model_id=str(mid),
            avg_lat=round(float(avg_lat_val), 2),
            std_lat=round(float(g_row['std_lat']), 2),
            avg_ttft=avg_ttft_val,
            avg_tps=round(float(g_row['avg_tps']) if not pd.isna(g_row['avg_tps']) else 0.0, 1),
            std_tps=round(float(g_row['std_tps']), 1),
            cv_tps=round(float(g_row['cv_tps']), 1),
            avg_tokens=round(float(g_row['avg_tokens'])) if not pd.isna(g_row['avg_tokens']) else 0,
            errors=int(g_row['errors']),
            runs=int(g_row['runs']),
            error_rate=float(g_row['error_rate']),
            avg_cost_per_1k=round(float(g_row['avg_cost_per_1k']), 6)
        )
        rows.append(row)

    rows.sort(key=lambda r: r.avg_tps, reverse=True)
    
    # 4. Composite Scores
    composite_scores = compute_radar_scores(rows)
    for row_obj in rows:
        row_obj.composite_score = composite_scores.get(row_obj.model_id, 0.0)

    return rows

def compute_radar_scores(rows: list[LeaderboardRow]) -> dict[str, float]:
    """
    Normalize the 5 radar dimensions and produce a weighted composite score.
    Weights: Speed 30% | Responsiveness 20% | Consistency 20% | Output 15% | Reliability 15%
    """
    if not rows:
        return {}

    raw: dict[str, dict[str, float]] = {}
    for row in rows:
        avg_tps = row.avg_tps if row.avg_tps > 0 else 0.01
        avg_ttft = row.avg_ttft if row.avg_ttft is not None and row.avg_ttft > 0 else 10.0
        cv = row.cv_tps if row.cv_tps > 0 else 100.0
        raw[row.model_id] = {
            "speed": avg_tps,
            "responsiveness": 1.0 / avg_ttft,
            "consistency": 1.0 / max(cv, 0.1),
            "output": float(row.avg_tokens),
            "reliability": max(0.0, 100.0 - row.error_rate),
        }

    axes = ["speed", "responsiveness", "consistency", "output", "reliability"]
    maxima = {axis: max(raw[m][axis] for m in raw) or 1.0 for axis in axes}
    weights = {"speed": 0.30, "responsiveness": 0.20, "consistency": 0.20, "output": 0.15, "reliability": 0.15}

    scores: dict[str, float] = {}
    for row in rows:
        vals = raw[row.model_id]
        score = sum(
            (vals[axis] / maxima[axis]) * 100.0 * weights[axis]
            for axis in axes
        )
        scores[row.model_id] = round(score, 1)

    return scores
