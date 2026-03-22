"""
visualization.py — Visualization Module (Hardware-Accelerated Edition)
──────────────────────────────────────────────────────────────────────
Plotly chart construction pipelines optimized for WebGL and batched rendering.

Architectural Enhancements:
  • WebGL GPU Acceleration: Uses go.Scattergl instead of standard SVG go.Scatter 
    to prevent client-side DOM freezing when visualizing dense metric clusters.
  • Batched DOM Mutation: Completely eliminates iterative fig.add_trace() loops 
    in favor of bulk go.Figure(data=[...]) trace instantiation.
  • High-Resolution Variance: Scatter charts now render the entire distribution 
    of TTFT vs TPS across all runs, instead of flattening to a single mean.
"""

from __future__ import annotations

import statistics as stats
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import (
    CHART_COLORS,
    LeaderboardRow,
)


# ── Type Aliases ────────────────────────────────────────────────────────────
_ModelAccumulator = dict[str, dict]


# ── Shared Layout Configuration ────────────────────────────────────────────

_FONT_FAMILY: str = "JetBrains Mono, monospace"
_FONT_SIZE: int = 11
_GRID_COLOR: str = "rgba(128,128,128,0.15)"
_BG_TRANSPARENT: str = "rgba(0,0,0,0)"


def _base_layout_kwargs(height: int = 380, bottom_margin: int = 50) -> dict:
    return {
        "height": height,
        "margin": dict(l=40, r=20, t=50, b=bottom_margin),
        "paper_bgcolor": _BG_TRANSPARENT,
        "plot_bgcolor": _BG_TRANSPARENT,
        "font": dict(family=_FONT_FAMILY, size=_FONT_SIZE),
    }


def _hex_to_rgba(hex_color: str, alpha: float = 0.1) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def color_for_index(i: int) -> str:
    return CHART_COLORS[i % len(CHART_COLORS)]


# ── Empty Figure Factory ───────────────────────────────────────────────────

def empty_figure(height: int = 80) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        height=height,
        paper_bgcolor=_BG_TRANSPARENT,
        plot_bgcolor=_BG_TRANSPARENT,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig


# ── Bar Chart: Latency / TTFT / Throughput ──────────────────────────────────

def build_bar_chart(model_stats: _ModelAccumulator) -> go.Figure:
    names, lat_vals, ttft_vals, tps_vals, cost_vals = [], [], [], [], []

    for mid, bucket in model_stats.items():
        if not bucket.get("latencies"):
            continue
        names.append(bucket["name"][:30])
        lat_vals.append(round(stats.mean(bucket["latencies"]), 2))
        ttft_vals.append(round(stats.mean(bucket["ttfts"]), 3) if bucket.get("ttfts") else 0.0)
        tps_vals.append(round(stats.mean(bucket["tps_vals"]), 1))

        total_costs = bucket.get("total_costs", [])
        comp_tokens = bucket.get("comp_tokens", [])
        avg_cost = stats.mean(total_costs) if total_costs else 0.0
        avg_tokens = stats.mean(comp_tokens) if comp_tokens else 0
        cost_per_1k = round(avg_cost / avg_tokens * 1000, 6) if avg_tokens > 0 else 0.0
        cost_vals.append(cost_per_1k)

    if not names:
        return empty_figure()

    # Batch trace construction
    traces = []
    for i, name in enumerate(names):
        c = color_for_index(i)

        traces.extend([
            go.Bar(x=[name], y=[lat_vals[i]], name=name, marker_color=c, showlegend=True, legendgroup=name, xaxis='x1', yaxis='y1'),
            go.Bar(x=[name], y=[ttft_vals[i]], name=name, marker_color=c, showlegend=False, legendgroup=name, xaxis='x2', yaxis='y2'),
            go.Bar(x=[name], y=[tps_vals[i]], name=name, marker_color=c, showlegend=False, legendgroup=name, xaxis='x3', yaxis='y3'),
            go.Bar(x=[name], y=[cost_vals[i]], name=name, marker_color=c, showlegend=False, legendgroup=name, xaxis='x4', yaxis='y4'),
        ])

    fig = make_subplots(
        rows=1, cols=4,
        subplot_titles=("Avg Latency (s) ↓", "Avg TTFT (s) ↓", "Avg tok/s ↑", "Cost/1K tokens ($) ↓"),
        horizontal_spacing=0.06,
    )
    fig.add_traces(traces)

    # Global hardware-accelerated state mutation
    fig.update_layout(
        **_base_layout_kwargs(height=380, bottom_margin=80),
        legend=dict(orientation="h", y=-0.25, x=0.5, xanchor="center"),
        barmode="group",
    )
    fig.update_xaxes(tickangle=45, gridcolor=_GRID_COLOR)
    fig.update_yaxes(gridcolor=_GRID_COLOR)

    return fig


# ── Scatter Chart: WebGL Accelerated Dense Clusters ─────────────────────────

def build_scatter_chart(model_stats: _ModelAccumulator) -> go.Figure:
    """
    Constructs a GPU-accelerated scatter plot mapping raw TTFT vs Throughput.
    Utilizes go.Scattergl to guarantee client rendering stability when mapping 
    thousands of data points from dense multi-run benchmarks.
    """
    traces = []

    for i, (mid, bucket) in enumerate(model_stats.items()):
        ttfts = bucket.get("ttfts", [])
        tps_vals = bucket.get("tps_vals", [])
        comp_tokens = bucket.get("comp_tokens", [])

        if not ttfts or not tps_vals:
            continue

        c = color_for_index(i)
        
        # Calculate dynamic bubble sizes natively without breaking vectorization
        bubble_sizes = [max(8, min(40, tok / 10)) for tok in comp_tokens] if comp_tokens else 12

        # ── WebGL Trace Instantiation ──
        traces.append(
            go.Scattergl(
                x=ttfts,
                y=tps_vals,
                mode="markers",
                marker=dict(
                    size=bubble_sizes,
                    color=c,
                    opacity=0.75,
                    line=dict(width=1, color="rgba(255, 255, 255, 0.4)"),
                ),
                name=bucket["name"][:30],
                hovertemplate=(
                    f"<b>{bucket['name'][:30]}</b><br>"
                    f"TTFT: %{{x:.3f}}s<br>"
                    f"tok/s: %{{y:.1f}}<br>"
                    f"<extra></extra>"
                ),
            )
        )

    if not traces:
        return empty_figure()

    # Bulk load payload into Figure to prevent DOM thrashing
    fig = go.Figure(data=traces)

    fig.update_layout(
        **_base_layout_kwargs(height=400),
        title="TTFT vs Throughput Variance (WebGL Accelerated)",
        xaxis_title="TTFT (s) →",
        yaxis_title="Throughput (tok/s) →",
        showlegend=True,
        legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center"),
    )
    fig.update_xaxes(gridcolor=_GRID_COLOR)
    fig.update_yaxes(gridcolor=_GRID_COLOR)

    return fig


# ── Consistency Chart: Batched Box Plots ────────────────────────────────────

def build_consistency_chart(model_stats: _ModelAccumulator) -> go.Figure:
    """
    Constructs variance distribution charts. Batches all Box traces in memory 
    before injecting them into the subplot grid to drastically reduce reflows.
    """
    traces_lat, traces_tps = [], []

    for i, (mid, bucket) in enumerate(model_stats.items()):
        if len(bucket.get("latencies", [])) < 2:
            continue

        c = color_for_index(i)
        display_name = bucket["name"][:25]

        traces_lat.append(
            go.Box(y=bucket["latencies"], name=display_name, marker_color=c, boxmean="sd", legendgroup=display_name, showlegend=True)
        )
        traces_tps.append(
            go.Box(y=bucket["tps_vals"], name=display_name, marker_color=c, boxmean="sd", legendgroup=display_name, showlegend=False)
        )

    if not traces_lat:
        return empty_figure()

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Latency Distribution (s)", "Throughput Distribution (tok/s)"),
        horizontal_spacing=0.1,
    )

    # Bulk trace injection
    fig.add_traces(traces_lat, rows=1, cols=1)
    fig.add_traces(traces_tps, rows=1, cols=2)

    fig.update_layout(
        **_base_layout_kwargs(height=380),
        legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center"),
    )
    fig.update_yaxes(gridcolor=_GRID_COLOR)

    return fig


# ── Radar Chart: 5-Axis Normalized Performance ─────────────────────────────

# ── Category Breakdown Chart ────────────────────────────────────────────────

def build_category_chart(breakdown_df: pd.DataFrame) -> Optional[go.Figure]:
    """
    Grouped bar chart showing avg tok/s per model per suite/preset category.

    Args:
        breakdown_df: Output of processing.build_category_breakdown().

    Returns:
        Plotly Figure, or None if no suite data is present.
    """
    if breakdown_df is None or breakdown_df.empty:
        return None

    labels = sorted(breakdown_df["suite_label"].unique())
    models = breakdown_df["model_name"].unique().tolist()

    traces = []
    for i, label in enumerate(labels):
        subset = breakdown_df[breakdown_df["suite_label"] == label]
        x_vals = subset["model_name"].tolist()
        y_vals = subset["avg_tps"].tolist()
        traces.append(
            go.Bar(
                x=x_vals,
                y=y_vals,
                name=label[:40],
                marker_color=color_for_index(i),
                hovertemplate="<b>%{x}</b><br>Category: " + label[:40] + "<br>tok/s: %{y:.1f}<extra></extra>",
            )
        )

    fig = go.Figure(data=traces)
    fig.update_layout(
        **_base_layout_kwargs(height=400, bottom_margin=100),
        title="Performance by Suite / Category",
        xaxis_title="Model",
        yaxis_title="Avg tok/s ↑",
        barmode="group",
        legend=dict(orientation="h", y=-0.35, x=0.5, xanchor="center"),
    )
    fig.update_xaxes(tickangle=45, gridcolor=_GRID_COLOR)
    fig.update_yaxes(gridcolor=_GRID_COLOR)

    return fig


# ── Drift Chart: Run-over-Run Performance ───────────────────────────────────

def build_drift_chart(model_stats: _ModelAccumulator) -> Optional[go.Figure]:
    """
    Line chart of tok/s per run index, one line per model.
    Reveals rate-limit throttling or server-side degradation across repeated calls.

    Returns:
        Plotly Figure, or None if all models have fewer than 2 runs.
    """
    traces = []

    for i, (mid, bucket) in enumerate(model_stats.items()):
        tps_vals = bucket.get("tps_vals", [])
        if len(tps_vals) < 2:
            continue

        c = color_for_index(i)
        run_indices = list(range(1, len(tps_vals) + 1))

        traces.append(
            go.Scatter(
                x=run_indices,
                y=[round(v, 1) for v in tps_vals],
                mode="lines+markers",
                name=bucket["name"][:30],
                line=dict(color=c, width=2),
                marker=dict(size=6, color=c),
                hovertemplate=(
                    f"<b>{bucket['name'][:30]}</b><br>"
                    f"Run: %{{x}}<br>"
                    f"tok/s: %{{y:.1f}}<br>"
                    f"<extra></extra>"
                ),
            )
        )

    if not traces:
        return None

    fig = go.Figure(data=traces)
    fig.update_layout(
        **_base_layout_kwargs(height=380, bottom_margin=60),
        title="Run-over-Run Throughput Drift",
        xaxis_title="Run #",
        yaxis_title="tok/s",
        xaxis=dict(tickmode="linear", dtick=1, gridcolor=_GRID_COLOR),
        yaxis=dict(gridcolor=_GRID_COLOR),
        legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center"),
    )

    return fig


_RADAR_AXES: list[str] = ["Speed", "Responsiveness", "Consistency", "Output Volume", "Reliability"]

def build_radar_chart(rows: list[LeaderboardRow]) -> go.Figure:
    if not rows:
        return empty_figure()

    raw_values = {}
    for row in rows:
        avg_tps = row.avg_tps if row.avg_tps > 0 else 0.01
        avg_ttft = row.avg_ttft if row.avg_ttft is not None and row.avg_ttft > 0 else 10.0
        cv = row.cv_tps if row.cv_tps > 0 else 100.0

        raw_values[row.model] = {
            "Speed": avg_tps,
            "Responsiveness": 1.0 / avg_ttft,
            "Consistency": 1.0 / max(cv, 0.1),
            "Output Volume": float(row.avg_tokens),
            "Reliability": max(0.0, 100.0 - row.error_rate),
        }

    axis_maxima = {
        axis: max([raw_values[m][axis] for m in raw_values]) or 1.0 
        for axis in _RADAR_AXES
    }

    traces = []
    for i, (model_name, vals) in enumerate(raw_values.items()):
        normalized = [round(vals[axis] / axis_maxima[axis] * 100.0, 1) for axis in _RADAR_AXES]
        normalized.append(normalized[0])  # Close polygon

        c = color_for_index(i)
        
        traces.append(
            go.Scatterpolar(
                r=normalized,
                theta=_RADAR_AXES + [_RADAR_AXES[0]],
                name=model_name[:30],
                fill="toself",
                fillcolor=_hex_to_rgba(c, alpha=0.1),
                line=dict(color=c, width=2),
                marker=dict(size=5),
            )
        )

    # Batched instantiation
    fig = go.Figure(data=traces)

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], gridcolor="rgba(128,128,128,0.2)"),
            angularaxis=dict(gridcolor="rgba(128,128,128,0.2)"),
            bgcolor=_BG_TRANSPARENT,
        ),
        height=450,
        margin=dict(l=60, r=60, t=40, b=40),
        paper_bgcolor=_BG_TRANSPARENT,
        font=dict(family=_FONT_FAMILY, size=_FONT_SIZE),
        legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center"),
        title="Normalized Performance Radar",
    )

    return fig
