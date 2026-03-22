"""
visualization.py — Visualization Facade
───────────────────────────────────────
Thin orchestrator re-exporting chart builders from chart_builders.py.
"""

from __future__ import annotations

from chart_builders import (
    empty_figure,
    build_bar_chart,
    build_scatter_chart,
    build_consistency_chart,
    build_category_chart,
    build_drift_chart,
    build_radar_chart,
)

__all__ = [
    "empty_figure",
    "build_bar_chart",
    "build_scatter_chart",
    "build_consistency_chart",
    "build_category_chart",
    "build_drift_chart",
    "build_radar_chart",
]
