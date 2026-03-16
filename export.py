"""
export.py — Export Module
─────────────────────────
Serialization pipelines for benchmark results and leaderboard data.

Responsibilities:
  • Serialize raw BenchmarkResult lists to CSV strings
  • Serialize raw BenchmarkResult lists to JSON strings
  • Build share-ready Markdown tables for Reddit/Discord
  • Write export strings to temporary files for Gradio download components

This module imports ONLY from `config` (internal), `csv`, `io`, `json`,
and `os`. It never imports Gradio, Pandas, Plotly, or requests.
"""

from __future__ import annotations

import csv
import io
import json
import os
from datetime import datetime, timezone
from typing import Optional

from config import (
    BenchmarkResult,
    LeaderboardRow,
)


# ── CSV Export ──────────────────────────────────────────────────────────────

_CSV_FIELDNAMES: list[str] = [
    "timestamp",
    "model_id",
    "model_name",
    "suite_label",
    "latency_sec",
    "ttft_sec",
    "tokens_per_sec",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "temperature",
    "top_p",
    "max_tokens_cfg",
    "error",
    "prompt",
    "response",
]


def export_results_csv(all_results: list[BenchmarkResult]) -> str:
    """
    Serialize a list of BenchmarkResult into a CSV string.

    Field order is fixed and documented in _CSV_FIELDNAMES.
    Missing fields default to empty string.

    Returns:
        Complete CSV content as a string (headers + rows).
    """
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=_CSV_FIELDNAMES)
    writer.writeheader()

    for result in all_results:
        row_dict = result.to_dict()
        writer.writerow(
            {field: row_dict.get(field, "") for field in _CSV_FIELDNAMES}
        )

    return buf.getvalue()


# ── JSON Export ─────────────────────────────────────────────────────────────

def export_results_json(all_results: list[BenchmarkResult]) -> str:
    """
    Serialize a list of BenchmarkResult into a pretty-printed JSON string.

    Returns:
        JSON array string with 2-space indentation.
    """
    return json.dumps(
        [result.to_dict() for result in all_results],
        indent=2,
        ensure_ascii=False,
    )


# ── Share-Ready Markdown ────────────────────────────────────────────────────

def build_share_markdown(
    rows: list[LeaderboardRow],
    insight_text: str,
) -> str:
    """
    Generate a Reddit/Discord-ready Markdown summary table with insights.

    Format:
      • Title with UTC timestamp
      • Ranked table with medals for top 3
      • Appended insight summary (reformatted from H2 to H3)
      • Attribution footer

    Args:
        rows: Sorted LeaderboardRow list from processing.build_leaderboard_rows().
        insight_text: Markdown insight string from insights.generate_insights().

    Returns:
        Complete Markdown string ready for clipboard copy.
    """
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")

    lines: list[str] = [
        "# OpenRouter Free Model Benchmark Results",
        f"*{now_str} UTC*\n",
        "| # | Model | Latency | TTFT | tok/s | Tokens | Errors |",
        "|---|-------|---------|------|-------|--------|--------|",
    ]

    for rank, row in enumerate(rows, 1):
        medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, str(rank))
        ttft_display = f"{row.avg_ttft}s" if row.avg_ttft is not None else "—"

        lines.append(
            f"| {medal} "
            f"| {row.model} "
            f"| {row.avg_lat}s "
            f"| {ttft_display} "
            f"| {row.avg_tps} "
            f"| {row.avg_tokens} "
            f"| {row.errors} |"
        )

    lines.append("")

    # Reformat insight heading level for embedding context
    reformatted_insight = insight_text.replace("## 🔎 ", "### ")
    lines.append(reformatted_insight)

    lines.append(
        "\n*Benchmarked with OpenRouter Free Model Benchmarker v3*"
    )

    return "\n".join(lines)


# ── File Writers ────────────────────────────────────────────────────────────

_EXPORT_DIR: str = "/tmp"


def write_export_files(
    csv_data: str,
    json_data: str,
) -> tuple[Optional[str], Optional[str]]:
    """
    Write CSV and JSON export strings to temporary files.

    Returns:
        Tuple of (csv_filepath, json_filepath).
        Either may be None if the corresponding data string is empty.
    """
    csv_path: Optional[str] = None
    json_path: Optional[str] = None

    if csv_data:
        csv_path = os.path.join(_EXPORT_DIR, "benchmark_results.csv")
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write(csv_data)

    if json_data:
        json_path = os.path.join(_EXPORT_DIR, "benchmark_results.json")
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(json_data)

    return csv_path, json_path
