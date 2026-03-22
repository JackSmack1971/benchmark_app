"""
state_managers.py — Gradio State & Business Logic Orchestration
─────────────────────────────────────────────────────────────
Handles all event callbacks, async dispatch, and UI state mutations.
Strictly decoupled from UI component declaration.
"""

from __future__ import annotations

import time
import asyncio
from datetime import datetime, timezone

import gradio as gr
import pandas as pd

from config import (
    ModelInfo,
    BenchmarkResult,
)
from network import (
    fetch_free_models,
    run_single_benchmark,
)
from processing import (
    build_leaderboard_dataframe,
    build_category_breakdown,
    build_sidebyside_markdown,
    estimate_tokens,
    resolve_preset_prompt,
    resolve_smart_defaults,
    resolve_suite_prompts,
    update_prompt_history,
    history_to_choices,
    apply_blind_labels,
    reveal_blind_results,
)
from aggregation import aggregate_stats, build_leaderboard_rows
from visualization import (
    empty_figure,
    build_bar_chart,
    build_scatter_chart,
    build_consistency_chart,
    build_radar_chart,
    build_category_chart,
    build_drift_chart,
)
from export import (
    export_results_csv,
    export_results_json,
    build_share_markdown,
    write_export_files,
)
from insights import generate_insights

# We import MAX_PARALLEL_WORKERS dynamically if needed, or straight from config
from config import MAX_PARALLEL_WORKERS

# ── Event Handlers (Async Orchestration) ────────────────────────────────────

def handle_key_status_update(key: str) -> str:
    return "✅ Key set" if key.strip() else "⚠️ No key"

async def handle_refresh_models(api_key: str) -> tuple:
    """Asynchronously fetch models without freezing the Gradio interface."""
    if not api_key.strip():
        return (gr.update(choices=[], value=[]), gr.update(choices=[], value=[]), "⚠️ Please set your API key first.", [], [])

    try:
        models = await fetch_free_models(api_key)
    except RuntimeError as exc:
        return (gr.update(choices=[], value=[]), gr.update(choices=[], value=[]), f"❌ Error: {exc}", [], [])

    providers = sorted(set(m.provider for m in models))
    model_choices = [(f"{m.name}  [{m.provider}]", m.id) for m in models]
    provider_choices = [(p.title(), p) for p in providers]

    return (gr.update(choices=model_choices, value=[]), gr.update(choices=provider_choices, value=[]),
            f"✅ {len(models)} free models · {len(providers)} providers", models, providers)

def handle_filter_by_provider(providers: list[str], cached_models: list[ModelInfo]) -> dict:
    if not providers:
        choices = [(f"{m.name}  [{m.provider}]", m.id) for m in cached_models]
    else:
        choices = [(f"{m.name}  [{m.provider}]", m.id) for m in cached_models if m.provider in providers]
    return gr.update(choices=choices, value=[])

def handle_model_info(selected_ids: list[str], cached_models: list[ModelInfo]) -> str:
    if not selected_ids: return "*Select models to see specs.*"
    parts: list[str] = []
    for sid in selected_ids:
        model = next((m for m in cached_models if m.id == sid), None)
        if model:
            parts.append(f"**{model.name}** · `{model.id}`\n> Context: **{model.context_length:,}** · Max output: **{model.max_completion:,}** · Modality: `{model.modality}`\n> {model.description}")
    return "\n\n".join(parts)

def handle_preset_change(preset_name: str) -> tuple[str, float, float]:
    return resolve_preset_prompt(preset_name), *resolve_smart_defaults(preset_name)

def handle_cancel(cancel_list: list[list[bool]]) -> dict:
    """Instantaneously mutates the boolean reference injected into async tasks."""
    for state in cancel_list:
        state[0] = True
    return gr.update(interactive=False, value="⏳ Cancelling...")

def handle_export(csv_data: str, json_data: str) -> tuple:
    return write_export_files(csv_data, json_data)

def normalize_suite(val: str) -> str | None:
    return val if val and val != "(none)" else None

_OUT_LOG, _OUT_INSIGHT, _OUT_LEADERBOARD, _OUT_BAR, _OUT_SCATTER, _OUT_CONSISTENCY, _OUT_RADAR, _OUT_CATEGORY, _OUT_DRIFT, _OUT_SIDEBYSIDE, _OUT_CSV, _OUT_JSON, _OUT_HISTORY, _OUT_SHARE_MD, _OUT_CANCEL_BTN = range(15)
_NUM_OUTPUTS = 15

def _empty_outputs(message: str, prompt_history: list[str]) -> tuple:
    ef = empty_figure()
    out = [""] * _NUM_OUTPUTS
    out[_OUT_LOG] = message
    out[_OUT_LEADERBOARD] = pd.DataFrame()
    out[_OUT_BAR] = out[_OUT_SCATTER] = out[_OUT_CONSISTENCY] = out[_OUT_RADAR] = out[_OUT_CATEGORY] = out[_OUT_DRIFT] = ef
    out[_OUT_HISTORY] = prompt_history
    out[_OUT_CANCEL_BTN] = gr.update(interactive=True, value="⛔ Cancel Benchmark")
    return tuple(out)

# ── Main Async Benchmark Orchestrator ───────────────────────────────────────

async def run_benchmark(
    api_key: str, selected_models: list[str], prompt_text: str, max_tokens: float,
    runs_per_model: float, parallel: bool, temperature: float, top_p: float,
    blind_mode: bool, suite_name: str | None, prompt_history: list[str],
    cached_models: list[ModelInfo], cancel_list: list[list[bool]],
    progress: gr.Progress = gr.Progress(track_tqdm=False),
):
    """
    Asynchronous benchmark generator. Coordinates parallel network requests natively
    via the asyncio event loop to prevent OS thread saturation.
    """
    cancel_flag = [False]
    cancel_list.clear()
    cancel_list.append(cancel_flag)

    if not api_key.strip(): yield _empty_outputs("⚠️ Set your API key first.", prompt_history); return
    if not selected_models: yield _empty_outputs("⚠️ Select at least one model.", prompt_history); return

    prompts_to_run = resolve_suite_prompts(suite_name) if suite_name else []
    if not prompts_to_run:
        if not prompt_text.strip(): yield _empty_outputs("⚠️ Enter a prompt or select a suite.", prompt_history); return
        prompts_to_run = [(prompt_text.strip(), "")]

    prompt_history = update_prompt_history(prompt_history, [p for p, _ in prompts_to_run])
    model_lookup = {m.id: m.name for m in cached_models}
    
    def name_for(mid: str) -> str:
        return apply_blind_labels(selected_models, mid) if blind_mode else model_lookup.get(mid, mid)

    runs, max_tok = int(runs_per_model), int(max_tokens)
    task_args = [(mid, prompt, label, run_idx) for prompt, label in prompts_to_run for mid in selected_models for run_idx in range(runs)]
    total_tasks = len(task_args)
    completed = 0
    all_results: list[BenchmarkResult] = []
    log_lines: list[str] = []
    start_wall = time.perf_counter()

    log_lines.append(f"### Benchmark Started — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")
    log_lines.append(f"Models: **{len(selected_models)}** · Prompts: **{len(prompts_to_run)}** · Runs/model: **{runs}** · Max tokens: **{max_tok}** · Temp: **{temperature}** · Top-p: **{top_p}** · {'🙈 Blind Mode' if blind_mode else '👁️ Open Mode'} · {'⚡ Parallel' if parallel else '📶 Sequential'}{f' | Suite: **{suite_name}**' if suite_name else ''}\n")

    ef = empty_figure()

    def make_progress_yield() -> tuple:
        eta_str = f" · ETA: **{((time.perf_counter() - start_wall) / completed) * (total_tasks - completed):.0f}s**" if completed > 0 else ""
        return ("\n".join(log_lines) + f"\n\n*Progress: {completed}/{total_tasks}{eta_str}*", "", pd.DataFrame(), ef, ef, ef, ef, ef, ef, "", "", "", prompt_history, "", gr.update(interactive=True, value="⛔ Cancel Benchmark"))

    async def execute_run(mid: str, prmpt: str, lbl: str, run_idx: int) -> BenchmarkResult:
        return await run_single_benchmark(api_key, mid, name_for(mid), prmpt, max_tok, temperature, top_p, cancel_flag, lbl)

    def log_result(res: BenchmarkResult, lbl: str, r_idx: int):
        status = "❌" if res.is_error else "✅"
        err = f" · ❌ {res.error[:50]}" if res.is_error else ""
        log_lines.append(f"{status} `{res.model_name}` run {r_idx + 1}{f' [{lbl}]' if lbl else ''}: **{res.latency_sec}s** · **{res.tokens_per_sec} tok/s**{err}")

    # ── Async Dispatch ──
    if parallel:
        sem = asyncio.Semaphore(min(MAX_PARALLEL_WORKERS, total_tasks))
        
        async def sem_task(args):
            async with sem:
                if cancel_flag[0]: return None
                return await execute_run(*args), args
                
        tasks = [asyncio.create_task(sem_task(a)) for a in task_args]
        
        for future in asyncio.as_completed(tasks):
            if cancel_flag[0]:
                log_lines.append("\n### ⛔ Cancelled by user")
                break
            
            res_tuple = await future
            if not res_tuple: continue
            
            result, args = res_tuple
            all_results.append(result)
            completed += 1
            log_result(result, args[2], args[3])
            progress(completed / total_tasks, desc=f"{completed}/{total_tasks}")
            yield make_progress_yield()
    else:
        for args in task_args:
            if cancel_flag[0]:
                log_lines.append("\n### ⛔ Cancelled by user")
                break
            result = await execute_run(*args)
            all_results.append(result)
            completed += 1
            log_result(result, args[2], args[3])
            progress(completed / total_tasks, desc=f"{completed}/{total_tasks}")
            yield make_progress_yield()

    cancel_list.clear()

    if not all_results: yield _empty_outputs("No results collected.", prompt_history); return

    if blind_mode: reveal_blind_results(all_results, model_lookup)

    model_stats = aggregate_stats(all_results)
    rows = build_leaderboard_rows(all_results)
    
    log_lines.append(f"\n### ✅ Complete{ ' (partial — cancelled)' if cancel_flag[0] else ''} — {round(time.perf_counter() - start_wall, 1)}s total — {len(all_results)} results")

    insights_md = generate_insights(rows, all_results)
    model_info_map = {m.id: m for m in cached_models} if cached_models else None
    category_breakdown = build_category_breakdown(all_results)
    category_fig = build_category_chart(category_breakdown) or empty_figure()
    drift_fig = build_drift_chart(model_stats) or empty_figure()
    yield (
        "\n".join(log_lines),
        insights_md,
        build_leaderboard_dataframe(rows, model_info_map),
        build_bar_chart(model_stats),
        build_scatter_chart(model_stats),
        build_consistency_chart(model_stats),
        build_radar_chart(rows),
        category_fig,
        drift_fig,
        build_sidebyside_markdown(model_stats),
        export_results_csv(all_results),
        export_results_json(all_results),
        prompt_history,
        build_share_markdown(rows, insights_md),
        gr.update(interactive=True, value="⛔ Cancel Benchmark"),
    )
