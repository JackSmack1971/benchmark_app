"""
visualization.py — Visualization Module
────────────────────────────────────────
Plotly chart construction pipelines for benchmark result presentation.

Responsibilities:
  • Build bar charts (latency, TTFT, throughput comparison)
  • Build scatter plots (TTFT vs throughput bubble chart)
  • Build consistency box plots (variance across multi-run benchmarks)
  • Build 5-axis radar charts (normalized performance profiles)
  • Provide a reusable empty figure factory for placeholder states

Design principles:
  • All data aggregation happens BEFORE entering this module (in processing.py)
  • Chart builders receive pre-computed dicts/lists and return native Figure objects
  • Gradio consumes returned Figures directly via gr.Plot — no serialization layer
  • Consistent theming via shared layout factory and CHART_COLORS palette

This module imports ONLY from `config` (internal), `statistics`, and `plotly`.
It never imports Gradio, Pandas, or requests.
"""

from __future__ import annotations

import statistics as stats
from typing import Optional

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import (
    CHART_COLORS,
    LeaderboardRow,
)


# ── Type Aliases ────────────────────────────────────────────────────────────

# Same accumulator shape produced by processing.aggregate_stats()
_ModelAccumulator = dict[str, dict]


# ── Shared Layout Configuration ────────────────────────────────────────────

_FONT_FAMILY: str = "JetBrains Mono, monospace"
_FONT_SIZE: int = 11
_GRID_COLOR: str = "rgba(128,128,128,0.15)"
_BG_TRANSPARENT: str = "rgba(0,0,0,0)"


def _base_layout_kwargs(
    height: int = 380,
    bottom_margin: int = 50,
) -> dict:
    """
    Return a reusable dict of layout properties applied to every chart.

    Centralizes font, background, margin, and grid styling so that
    individual chart builders only specify their unique properties.
    """
    return {
        "height": height,
        "margin": dict(l=40, r=20, t=50, b=bottom_margin),
        "paper_bgcolor": _BG_TRANSPARENT,
        "plot_bgcolor": _BG_TRANSPARENT,
        "font": dict(family=_FONT_FAMILY, size=_FONT_SIZE),
    }


def _apply_grid(fig: go.Figure) -> None:
    """Apply consistent grid styling to all axes in a figure."""
    fig.update_xaxes(gridcolor=_GRID_COLOR)
    fig.update_yaxes(gridcolor=_GRID_COLOR)


def _hex_to_rgba(hex_color: str, alpha: float = 0.1) -> str:
    """Convert '#3b82f6' → 'rgba(59,130,246,0.1)' for fill colors."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def color_for_index(i: int) -> str:
    """Cycle through the palette deterministically."""
    return CHART_COLORS[i % len(CHART_COLORS)]


# ── Empty Figure Factory ───────────────────────────────────────────────────

def empty_figure(height: int = 80) -> go.Figure:
    """
    Return a minimal transparent Figure used as a placeholder
    in Gradio gr.Plot components before data is available.
    """
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
    """
    Three-panel grouped bar chart comparing:
      • Average Latency (s) — lower is better
      • Average TTFT (s) — lower is better
      • Average tok/s — higher is better

    Each model gets a consistent color across all three subplots
    via legendgroup linkage.

    Args:
        model_stats: Per-model accumulator from processing.aggregate_stats().

    Returns:
        plotly.graph_objects.Figure ready for gr.Plot consumption.
    """
    names: list[str] = []
    lat_vals: list[float] = []
    ttft_vals: list[float] = []
    tps_vals: list[float] = []

    for mid, bucket in model_stats.items():
        if not bucket["latencies"]:
            continue
        names.append(bucket["name"][:30])
        lat_vals.append(round(stats.mean(bucket["latencies"]), 2))
        ttft_vals.append(
            round(stats.mean(bucket["ttfts"]), 3)
            if bucket["ttfts"] else 0.0
        )
        tps_vals.append(round(stats.mean(bucket["tps_vals"]), 1))

    if not names:
        return empty_figure()

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=("Avg Latency (s) ↓", "Avg TTFT (s) ↓", "Avg tok/s ↑"),
        horizontal_spacing=0.08,
    )

    for i, name in enumerate(names):
        c = color_for_index(i)
        legend_group = name

        fig.add_trace(
            go.Bar(
                x=[name],
                y=[lat_vals[i]],
                name=name,
                marker_color=c,
                showlegend=True,
                legendgroup=legend_group,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Bar(
                x=[name],
                y=[ttft_vals[i]],
                name=name,
                marker_color=c,
                showlegend=False,
                legendgroup=legend_group,
            ),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Bar(
                x=[name],
                y=[tps_vals[i]],
                name=name,
                marker_color=c,
                showlegend=False,
                legendgroup=legend_group,
            ),
            row=1,
            col=3,
        )

    fig.update_layout(
        **_base_layout_kwargs(height=380, bottom_margin=80),
        legend=dict(orientation="h", y=-0.25, x=0.5, xanchor="center"),
        barmode="group",
    )
    fig.update_xaxes(tickangle=45, gridcolor=_GRID_COLOR)
    fig.update_yaxes(gridcolor=_GRID_COLOR)

    return fig


# ── Scatter Chart: TTFT vs Throughput ───────────────────────────────────────

def build_scatter_chart(model_stats: _ModelAccumulator) -> go.Figure:
    """
    Bubble scatter plot mapping:
      • X-axis: Average TTFT (seconds)
      • Y-axis: Average tok/s
      • Bubble size: proportional to average completion token count

    Ideal models cluster in the lower-left (fast TTFT) and upper region
    (high throughput). Bubble size adds a third dimension for output volume.

    Args:
        model_stats: Per-model accumulator from processing.aggregate_stats().

    Returns:
        plotly.graph_objects.Figure ready for gr.Plot consumption.
    """
    fig = go.Figure()

    trace_count = 0
    for i, (mid, bucket) in enumerate(model_stats.items()):
        if not bucket["latencies"]:
            continue

        avg_ttft = (
            stats.mean(bucket["ttfts"]) if bucket["ttfts"] else 0.0
        )
        avg_tps = (
            stats.mean(bucket["tps_vals"]) if bucket["tps_vals"] else 0.0
        )
        avg_tok = (
            stats.mean(bucket["comp_tokens"]) if bucket["comp_tokens"] else 10.0
        )

        c = color_for_index(i)
        display_name = bucket["name"][:25]

        bubble_size = max(12, min(60, avg_tok / 8))

        fig.add_trace(
            go.Scatter(
                x=[round(avg_ttft, 3)],
                y=[round(avg_tps, 1)],
                mode="markers+text",
                marker=dict(
                    size=bubble_size,
                    color=c,
                    opacity=0.8,
                    line=dict(width=1, color="white"),
                ),
                text=[display_name],
                textposition="top center",
                textfont=dict(size=10),
                name=bucket["name"][:30],
                hovertemplate=(
                    f"<b>{bucket['name']}</b><br>"
                    f"TTFT: %{{x:.3f}}s<br>"
                    f"tok/s: %{{y:.1f}}<br>"
                    f"Tokens: {avg_tok:.0f}"
                    f"<extra></extra>"
                ),
            )
        )
        trace_count += 1

    if trace_count == 0:
        return empty_figure()

    fig.update_layout(
        **_base_layout_kwargs(height=400),
        title="TTFT vs Throughput (bubble = output size)",
        xaxis_title="Avg TTFT (s) →",
        yaxis_title="Avg tok/s →",
        showlegend=False,
    )
    _apply_grid(fig)

    return fig


# ── Consistency Chart: Box Plots ────────────────────────────────────────────

def build_consistency_chart(model_stats: _ModelAccumulator) -> go.Figure:
    """
    Dual-panel box plot showing distribution spread for:
      • Latency (seconds) — left panel
      • Throughput (tok/s) — right panel

    Requires ≥ 2 runs per model to produce meaningful distributions.
    Models with only 1 run are silently excluded.

    Box plots include:
      • Median line
      • Mean marker (boxmean="sd" shows mean + standard deviation band)
      • Whiskers at 1.5 × IQR
      • Outlier points

    Args:
        model_stats: Per-model accumulator from processing.aggregate_stats().

    Returns:
        plotly.graph_objects.Figure ready for gr.Plot consumption.
    """
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Latency Distribution (s)",
            "Throughput Distribution (tok/s)",
        ),
        horizontal_spacing=0.1,
    )

    trace_count = 0
    for i, (mid, bucket) in enumerate(model_stats.items()):
        if len(bucket["latencies"]) < 2:
            continue

        c = color_for_index(i)
        display_name = bucket["name"][:25]
        legend_group = display_name

        fig.add_trace(
            go.Box(
                y=bucket["latencies"],
                name=display_name,
                marker_color=c,
                boxmean="sd",
                legendgroup=legend_group,
                showlegend=True,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Box(
                y=bucket["tps_vals"],
                name=display_name,
                marker_color=c,
                boxmean="sd",
                legendgroup=legend_group,
                showlegend=False,
            ),
            row=1,
            col=2,
        )
        trace_count += 1

    if trace_count == 0:
        return empty_figure()

    fig.update_layout(
        **_base_layout_kwargs(height=380),
        legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center"),
    )
    fig.update_yaxes(gridcolor=_GRID_COLOR)

    return fig


# ── Radar Chart: 5-Axis Normalized Performance ─────────────────────────────

_RADAR_AXES: list[str] = [
    "Speed",
    "Responsiveness",
    "Consistency",
    "Output Volume",
    "Reliability",
]


def build_radar_chart(rows: list[LeaderboardRow]) -> go.Figure:
    """
    5-axis radar chart providing a normalized performance fingerprint
    for each model:

      • Speed:         1 / avg_latency (higher = faster)
      • Responsiveness: 1 / avg_ttft (higher = faster first token)
      • Consistency:    1 / CV% (higher = more stable throughput)
      • Output Volume:  avg_tokens (higher = more verbose)
      • Reliability:    100 - error_rate% (higher = fewer failures)

    All axes are normalized to 0–100 relative to the best-performing
    model on each axis. This creates a fair comparison profile regardless
    of absolute scale differences between metrics.

    The polygon is closed by appending the first value at the end.

    Args:
        rows: Sorted LeaderboardRow list from processing.build_leaderboard_rows().

    Returns:
        plotly.graph_objects.Figure ready for gr.Plot consumption.
    """
    if not rows:
        return empty_figure()

    # ── Step 1: Compute raw axis values per model ───────────────────────
    raw_values: dict[str, dict[str, float]] = {}

    for row in rows:
        avg_tps = row.avg_tps if row.avg_tps > 0 else 0.01
        avg_ttft = (
            row.avg_ttft
            if row.avg_ttft is not None and row.avg_ttft > 0
            else 10.0
        )
        cv = row.cv_tps if row.cv_tps > 0 else 100.0

        raw_values[row.model] = {
            "Speed": avg_tps,
            "Responsiveness": 1.0 / avg_ttft,
            "Consistency": 1.0 / max(cv, 0.1),
            "Output Volume": float(row.avg_tokens),
            "Reliability": max(0.0, 100.0 - row.error_rate),
        }

    # ── Step 2: Find per-axis maximums for normalization ────────────────
    axis_maxima: dict[str, float] = {}
    for axis in _RADAR_AXES:
        vals = [raw_values[m][axis] for m in raw_values]
        axis_maxima[axis] = max(vals) if max(vals) > 0 else 1.0

    # ── Step 3: Build traces ────────────────────────────────────────────
    fig = go.Figure()

    for i, (model_name, vals) in enumerate(raw_values.items()):
        normalized = [
            round(vals[axis] / axis_maxima[axis] * 100.0, 1)
            for axis in _RADAR_AXES
        ]
        # Close the polygon
        normalized.append(normalized[0])

        c = color_for_index(i)
        fill_color = _hex_to_rgba(c, alpha=0.1)

        fig.add_trace(
            go.Scatterpolar(
                r=normalized,
                theta=_RADAR_AXES + [_RADAR_AXES[0]],
                name=model_name[:30],
                fill="toself",
                fillcolor=fill_color,
                line=dict(color=c, width=2),
                marker=dict(size=5),
            )
        )

    # ── Step 4: Layout ──────────────────────────────────────────────────
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                gridcolor="rgba(128,128,128,0.2)",
            ),
            angularaxis=dict(
                gridcolor="rgba(128,128,128,0.2)",
            ),
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
