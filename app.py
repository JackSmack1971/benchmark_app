"""
app.py — Gradio UI Shell (Async Edition)
────────────────────────────────────────
Thin presentation and routing layer orchestrating via asyncio.

Responsibilities:
  • Define the Gradio Blocks layout
  • Wire UI events to purely asynchronous network fetch/execution handlers
  • Dispatch parallel tasks natively via asyncio.gather/as_completed
  • Protect the main thread from blocking operations
"""

from __future__ import annotations

import os
import time
import asyncio
from datetime import datetime, timezone

import gradio as gr
import pandas as pd

from config import (
    BENCHMARK_PRESETS,
    BENCHMARK_SUITES,
    METRIC_GLOSSARY,
    CSS,
    MAX_PARALLEL_WORKERS,
    ModelInfo,
    BenchmarkResult,
)
from network import (
    fetch_free_models,
    run_single_benchmark,
)
from processing import (
    aggregate_stats,
    build_leaderboard_rows,
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
    rows = build_leaderboard_rows(model_stats)
    
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


# ── Gradio Blocks Layout ───────────────────────────────────────────────────

def build_app() -> gr.Blocks:
    default_api_key = os.environ.get("OPENROUTER_API_KEY", "")

    with gr.Blocks(title="OpenRouter Free Model Benchmarker v3", css=CSS) as app:
        prompt_history_state = gr.State([])
        csv_state = gr.State("")
        json_state = gr.State("")
        share_md_state = gr.State("")
        model_cache_state = gr.State([])
        provider_cache_state = gr.State([])
        cancel_state = gr.State([])

        gr.Markdown("# ⚡ OpenRouter Free Model Benchmarker v3\n**Discover → Filter → Configure → Benchmark → Analyze → Share**", elem_classes="header-block")

        with gr.Accordion("🔑 API Key", open=not bool(default_api_key)):
            with gr.Row():
                api_key_input = gr.Textbox(label="OpenRouter API Key", type="password", placeholder="sk-or-v1-...", value=default_api_key, info="Free tier works — https://openrouter.ai/settings/keys", scale=3)
                key_status = gr.Textbox(label="Status", interactive=False, scale=1, value="✅ Loaded from env" if default_api_key else "⚠️ No key")
            api_key_input.change(handle_key_status_update, inputs=[api_key_input], outputs=[key_status])

        gr.Markdown("---\n### 🔍 Model Discovery")
        with gr.Row():
            provider_filter = gr.Dropdown(label="Filter by Provider", choices=[], multiselect=True, info="e.g. google, meta-llama, mistralai", scale=1)
            refresh_btn = gr.Button("🔄 Discover Free Models", variant="primary", size="lg", scale=1)
            model_status = gr.Markdown("*Click discover to fetch models*")

        model_selector = gr.Dropdown(label="Select Models to Benchmark", choices=[], multiselect=True, allow_custom_value=False, info="Multi-select — choose 2+ for comparison")
        model_info_display = gr.Markdown("")

        refresh_btn.click(handle_refresh_models, inputs=[api_key_input], outputs=[model_selector, provider_filter, model_status, model_cache_state, provider_cache_state])
        provider_filter.change(handle_filter_by_provider, inputs=[provider_filter, model_cache_state], outputs=[model_selector])
        model_selector.change(handle_model_info, inputs=[model_selector, model_cache_state], outputs=[model_info_display])

        gr.Markdown("---\n### ⚙️ Configuration")
        with gr.Row():
            with gr.Column(scale=3):
                prompt_input = gr.Textbox(label="Benchmark Prompt", lines=4, placeholder="Enter your prompt, pick a preset, or use a suite →")
                token_counter = gr.Markdown("~0 tokens", elem_classes="token-counter")
            with gr.Column(scale=1):
                suite_dropdown = gr.Dropdown(label="🧪 Benchmark Suite", choices=["(none)"] + list(BENCHMARK_SUITES.keys()), value="(none)", info="Overrides single prompt — runs multiple presets")
                preset_dropdown = gr.Dropdown(label="Single Preset", choices=list(BENCHMARK_PRESETS.keys()), value=None)
                history_dropdown = gr.Dropdown(label="Prompt History", choices=[], value=None, info="Session history")

        prompt_input.change(estimate_tokens, prompt_input, token_counter)
        history_dropdown.change(lambda x: x if x else "", inputs=[history_dropdown], outputs=[prompt_input])

        with gr.Row():
            max_tokens_slider = gr.Slider(label="Max Tokens", minimum=32, maximum=4096, value=512, step=32)
            runs_slider = gr.Slider(label="Runs / Model", minimum=1, maximum=10, value=1, step=1, info="↑ = better variance data")

        with gr.Row():
            temperature_slider = gr.Slider(label="Temperature", minimum=0.0, maximum=2.0, value=0.7, step=0.05)
            top_p_slider = gr.Slider(label="Top-p", minimum=0.0, maximum=1.0, value=1.0, step=0.05)
            parallel_check = gr.Checkbox(label="⚡ Parallel", value=True)
            blind_mode_check = gr.Checkbox(label="🙈 Blind Mode", value=False, info="Hide names during run")

        preset_dropdown.change(handle_preset_change, inputs=[preset_dropdown], outputs=[prompt_input, temperature_slider, top_p_slider])

        with gr.Row():
            run_btn = gr.Button("🚀 Run Benchmark", variant="primary", size="lg", scale=3)
            cancel_btn = gr.Button("⛔ Cancel Benchmark", variant="stop", size="lg", scale=1)

        gr.Markdown("---\n### 📊 Results")
        insight_output = gr.Markdown("", elem_classes="insight-box")

        with gr.Tabs():
            with gr.Tab("📋 Leaderboard"): leaderboard_df_output = gr.Dataframe(label="Sortable — click column headers to re-rank", interactive=False, wrap=True)
            with gr.Tab("📊 Charts"): bar_chart_output = gr.Plot(label="Metric Comparison"); scatter_chart_output = gr.Plot(label="TTFT vs Throughput")
            with gr.Tab("🎯 Radar"): radar_chart_output = gr.Plot(label="Normalized 5-Axis Comparison")
            with gr.Tab("📈 Consistency"): consistency_chart_output = gr.Plot(label="Variance (needs ≥2 runs/model)")
            with gr.Tab("🗂️ By Category"): category_chart_output = gr.Plot(label="tok/s by Suite / Category (run a suite to populate)")
            with gr.Tab("📉 Drift"): drift_chart_output = gr.Plot(label="Run-over-Run Throughput (needs ≥2 runs/model)")
            with gr.Tab("🔀 Side-by-Side"): sidebyside_output = gr.Markdown("*Run with 2+ models to compare outputs.*")
            with gr.Tab("📝 Live Log"): log_output = gr.Markdown("*Waiting for benchmark...*")
            with gr.Tab("💾 Export & Share"):
                gr.Markdown("**Download** raw data or **copy** a share-ready summary:")
                with gr.Row(): csv_download = gr.File(label="CSV", interactive=False); json_download = gr.File(label="JSON", interactive=False)
                export_btn = gr.Button("📥 Generate Export Files", variant="secondary")
                gr.Markdown("---")
                share_md_output = gr.Code(label="📋 Share-Ready Markdown", language="markdown", lines=12, interactive=False)

        with gr.Accordion("📖 Metric Glossary", open=False):
            gr.Markdown(METRIC_GLOSSARY, elem_classes="metric-glossary")

        export_btn.click(handle_export, inputs=[csv_state, json_state], outputs=[csv_download, json_download])
        cancel_btn.click(handle_cancel, inputs=[cancel_state], outputs=[cancel_btn])

        run_btn.click(normalize_suite, inputs=[suite_dropdown], outputs=[suite_dropdown]).then(
            run_benchmark,
            inputs=[api_key_input, model_selector, prompt_input, max_tokens_slider, runs_slider, parallel_check, temperature_slider, top_p_slider, blind_mode_check, suite_dropdown, prompt_history_state, model_cache_state, cancel_state],
            outputs=[log_output, insight_output, leaderboard_df_output, bar_chart_output, scatter_chart_output, consistency_chart_output, radar_chart_output, category_chart_output, drift_chart_output, sidebyside_output, csv_state, json_state, prompt_history_state, share_md_state, cancel_btn]
        ).then(lambda history: gr.update(choices=history_to_choices(history), value=None), inputs=[prompt_history_state], outputs=[history_dropdown]
        ).then(lambda md: md, inputs=[share_md_state], outputs=[share_md_output]
        ).then(lambda: gr.update(interactive=True, value="⛔ Cancel Benchmark"), outputs=[cancel_btn])

        gr.Markdown("---\n<center>v3 · Radar · Blind Mode · Suites · Cancel · Smart Defaults · <a href='https://openrouter.ai' target='_blank'>OpenRouter</a></center>")

    return app
