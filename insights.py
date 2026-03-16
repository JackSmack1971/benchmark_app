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

from config import LeaderboardRow


def generate_insights(rows: list[LeaderboardRow]) -> str:
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

    Returns:
        Markdown-formatted string. Empty string if no rows provided.
    """
    if not rows:
        return ""

    sections: list[str] = ["## 🔎 Insight Summary\n"]

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

    return "\n\n".join(sections)
