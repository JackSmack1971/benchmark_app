"""
insights.py — Insight Generation Engine
────────────────────────────────────────
Auto-generates natural-language analysis from aggregated benchmark data.

Responsibilities:
  • Identify fastest throughput, lowest latency, best TTFT models
  • Detect consistency leaders (lowest CV%)
  • Compare output volume spread across models
  • Flag models with errors
  • Produce a Markdown-formatted summary for the UI insight box

This module imports ONLY from `config` (internal). It never imports
Gradio, Pandas, Plotly, requests, or statistics.
"""

from __future__ import annotations

import re
from typing import Optional

from config import LeaderboardRow, BenchmarkResult


def classify_errors(results: list[BenchmarkResult]) -> dict[str, dict[str, int]]:
    """
    Classify error strings per model into actionable categories.

    Categories:
        rate_limit   — HTTP 429 or "rate limit" text
        server_error — HTTP 5xx codes
        timeout      — timeout / timed out
        empty        — empty response or blank output
        other        — catch-all

    Returns:
        {model_name: {category: count, ...}}
        Only models with at least one error are included.
    """
    taxonomy: dict[str, dict[str, int]] = {}

    for r in results:
        if not r.is_error:
            continue

        err = (r.error or "").lower()
        name = r.model_name

        if name not in taxonomy:
            taxonomy[name] = {"rate_limit": 0, "server_error": 0, "timeout": 0, "empty": 0, "other": 0}

        if "429" in err or "rate limit" in err:
            taxonomy[name]["rate_limit"] += 1
        elif re.search(r"\b5\d{2}\b", err):
            taxonomy[name]["server_error"] += 1
        elif "timeout" in err or "timed out" in err:
            taxonomy[name]["timeout"] += 1
        elif "empty" in err or not r.response.strip():
            taxonomy[name]["empty"] += 1
        else:
            taxonomy[name]["other"] += 1

    return taxonomy


def generate_insights(rows: list[LeaderboardRow], results: Optional[list[BenchmarkResult]] = None) -> str:
    """
    Auto-generate a natural-language insight summary from leaderboard data.

    Analyzes the sorted LeaderboardRow list to identify:
      • Highest throughput model
      • Lowest latency model (if different from throughput leader)
      • Fastest Time To First Token
      • Most consistent model (lowest CV%, requires multi-run data)
      • Output volume spread between most and least verbose models
      • Models with error occurrences

    Args:
        rows: Sorted LeaderboardRow list from processing.build_leaderboard_rows().
              Expected to be sorted descending by avg_tps.
        results: Optional raw BenchmarkResult list for error taxonomy analysis.

    Returns:
        Markdown-formatted string. Empty string if no rows provided.
    """
    if not rows:
        return ""

    sections: list[str] = ["## 🔎 Insight Summary\n"]

    # ── Overall recommended model (highest composite score) ─────────────
    best = max(rows, key=lambda r: r.composite_score)
    sections.append(
        f"🏆 **Recommended:** {best.model} "
        f"(composite score **{best.composite_score}/100**)"
    )

    # ── Fastest throughput (rows[0] since sorted desc by avg_tps) ───────
    fastest = rows[0]
    sections.append(
        f"**Fastest throughput:** {fastest.model} "
        f"at **{fastest.avg_tps} tok/s**"
    )

    # ── Lowest latency (may differ from throughput leader) ──────────────
    by_latency = sorted(rows, key=lambda r: r.avg_lat)
    latency_leader = by_latency[0]
    if latency_leader.model != fastest.model:
        sections.append(
            f"**Lowest latency:** {latency_leader.model} "
            f"at **{latency_leader.avg_lat}s**"
        )

    # ── Best TTFT ───────────────────────────────────────────────────────
    with_ttft = [r for r in rows if r.avg_ttft is not None]
    if with_ttft:
        ttft_leader = min(with_ttft, key=lambda r: r.avg_ttft)
        sections.append(
            f"**Fastest first token:** {ttft_leader.model} "
            f"at **{ttft_leader.avg_ttft}s TTFT**"
        )

    # ── Most consistent (lowest CV%, requires multi-run) ────────────────
    with_cv = [r for r in rows if r.cv_tps > 0]
    if with_cv:
        consistency_leader = min(with_cv, key=lambda r: r.cv_tps)
        sections.append(
            f"**Most consistent:** {consistency_leader.model} "
            f"with **{consistency_leader.cv_tps}% CV** in throughput"
        )

    # ── Output volume spread ────────────────────────────────────────────
    if len(rows) > 1:
        most_verbose = max(rows, key=lambda r: r.avg_tokens)
        least_verbose = min(rows, key=lambda r: r.avg_tokens)

        # Only report if there's a meaningful difference (≥30%)
        if (
            least_verbose.avg_tokens > 0
            and most_verbose.avg_tokens > least_verbose.avg_tokens * 1.3
        ):
            sections.append(
                f"**Output spread:** {most_verbose.model} produced "
                f"~{most_verbose.avg_tokens} tokens vs "
                f"{least_verbose.model}'s ~{least_verbose.avg_tokens}"
            )

    # ── Error warnings ──────────────────────────────────────────────────
    error_models = [r for r in rows if r.errors > 0]
    if error_models:
        names_with_counts = ", ".join(
            f"{r.model} ({r.errors})" for r in error_models
        )
        sections.append(f"**⚠️ Errors detected:** {names_with_counts}")

    # ── Error taxonomy breakdown ────────────────────────────────────────
    if results:
        taxonomy = classify_errors(results)
        if taxonomy:
            table_rows = ["| Model | Rate Limits | Timeouts | Server Errors | Empty | Other |",
                          "|:------|------------:|---------:|--------------:|------:|------:|"]
            for model_name, counts in taxonomy.items():
                table_rows.append(
                    f"| {model_name} "
                    f"| {counts['rate_limit']} "
                    f"| {counts['timeout']} "
                    f"| {counts['server_error']} "
                    f"| {counts['empty']} "
                    f"| {counts['other']} |"
                )
            sections.append("**⚠️ Failure Breakdown**\n\n" + "\n".join(table_rows))

    return "\n\n".join(sections)
