import pandas as pd
from typing import Optional
from config import BenchmarkResult, LeaderboardRow

# ── Strict PyArrow Engine Enforcement ───────────────────────────────────────
pd.options.mode.copy_on_write = True
pd.options.mode.dtype_backend = "pyarrow"

def process_results_vectorized(all_results: list[BenchmarkResult]) -> pd.DataFrame:
    """
    Ingests raw results directly into zero-copy Arrow memory and performs 
    vectorized aggregations, eliminating native Python scalar loops.
    """
    if not all_results:
        return pd.DataFrame()

    # Immediate DataFrame ingestion
    df = pd.DataFrame([r.to_dict() for r in all_results])

    # Method chaining for transformations and aggregations
    return (
        df
        .assign(
            is_error=lambda x: x['error'].notna()
        )
        .groupby(['model_id', 'model_name'], as_index=False)
        .agg(
            runs=('model_id', 'count'),
            errors=('is_error', 'sum'),
            avg_lat=('latency_sec', 'mean'),
            std_lat=('latency_sec', 'std'),
            avg_ttft=('ttft_sec', 'mean'),
            avg_tps=('tokens_per_sec', 'mean'),
            std_tps=('tokens_per_sec', 'std'),
            avg_tokens=('completion_tokens', 'mean')
        )
        .assign(
            # Handle NaN standard deviations for single-run models
            std_lat=lambda x: x['std_lat'].fillna(0.0),
            std_tps=lambda x: x['std_tps'].fillna(0.0),
            
            # Vectorized coefficient of variation (CV%)
            cv_tps=lambda x: (x['std_tps'] / x['avg_tps'] * 100.0).fillna(0.0),
            
            # Error rate calculation
            error_rate=lambda x: (x['errors'] / x['runs'] * 100.0).round(1),
            
            # Rounding for display
            avg_lat=lambda x: x['avg_lat'].round(2),
            std_lat=lambda x: x['std_lat'].round(2),
            avg_ttft=lambda x: x['avg_ttft'].round(3),
            avg_tps=lambda x: x['avg_tps'].round(1),
            std_tps=lambda x: x['std_tps'].round(1),
            cv_tps=lambda x: x['cv_tps'].round(1),
            avg_tokens=lambda x: x['avg_tokens'].round(0).astype('int64[pyarrow]')
        )
        .sort_values(by='avg_tps', ascending=False)
        .reset_index(drop=True)
    )
