"""
app.py — Gradio UI Shell
─────────────────────────
Thin presentation and routing layer. Owns zero business logic.

Responsibilities:
  • Define the Gradio Blocks layout (rows, columns, tabs, accordions)
  • Wire UI events to domain-module functions
  • Manage gr.State for session-scoped variables
  • Provide gr.Progress feedback during benchmark runs
  • Orchestrate the benchmark execution loop (parallel/sequential dispatch)

All data fetching, aggregation, visualization, export, and insight
generation are delegated to their respective domain modules:
  • network.py    — HTTP I/O, model discovery, streaming benchmarks
  • processing.py — Pandas aggregation, stats, token estimation
  • visualization.py — Plotly chart construction
  • export.py     — CSV, JSON, Markdown serialization
  • insights.py   — Natural-language summary engine

Architecture contract:
  Requests (network) → Pandas (processing) → Plotly (visualization) → Gradio output
"""

from __future__ import annotations

import time
import concurrent.futures
from datetime import datetime, timezone

import gradio as gr
import pandas as pd

from config import (
    OPENROUTER_API_KEY,
    BENCHMARK_PRESETS,
    BENCHMARK_SUITES,
    METRIC_GLOSSARY,
    CSS,
    MAX_PARALLEL_WORKERS,
    ModelInfo,
    BenchmarkResult,
)
from network import (
    set_api_key,
    get_api_key,
    request_cancel,
    reset_cancel,
    is_cancelled,
    fetch_free_models,
    run_single_benchmark,
)
from processing import (
    aggregate_stats,
    build_leaderboard_rows,
    build_leaderboard_dataframe,
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
)
from export import (
    export_results_csv,
    export_results_json,
    build_share_markdown,
    write_export_files,
)
from insights import generate_insights


# ── Module-Level Model Cache ────────────────────────────────────────────────
# Populated by refresh_models(), read by event handlers.
# Not shared across users — Gradio 6+ queues requests per-session.

_cached_models: list[ModelInfo] = []
_all_providers: list[str] = []


# ── Event Handlers (thin wrappers delegating to domain modules) ─────────────

def handle_set_api_key(key: str) -> str:
    """Delegate API key storage to network module."""
    set_api_key(key)
    return "✅ Key set" if key.strip() else "⚠️ No key"


def handle_refresh_models() -> tuple:
    """
    Fetch free models via network module, populate module cache,
    and return Gradio component updates for model selector + provider filter.
    """
    global _cached_models, _all_providers

    try:
        _cached_models = fetch_free_models()
    except RuntimeError as exc:
        return (
            gr.update(choices=[], value=[]),
            gr.update(choices=[], value=[]),
            f"❌ Error: {exc}",
        )

    _all_providers = sorted(set(m.provider for m in _cached_models))

    model_choices = [
        (f"{m.name}  [{m.provider}]", m.id) for m in _cached_models
    ]
    provider_choices = [
        (p.title(), p) for p in _all_providers
    ]

    return (
        gr.update(choices=model_choices, value=[]),
        gr.update(choices=provider_choices, value=[]),
        f"✅ {len(_cached_models)} free models · {len(_all_providers)} providers",
    )


def handle_filter_by_provider(providers: list[str]) -> dict:
    """Filter model selector choices by selected providers."""
    if not providers:
        choices = [
            (f"{m.name}  [{m.provider}]", m.id) for m in _cached_models
        ]
    else:
        choices = [
            (f"{m.name}  [{m.provider}]", m.id)
            for m in _cached_models
            if m.provider in providers
        ]
    return gr.update(choices=choices, value=[])


def handle_model_info(selected_ids: list[str]) -> str:
    """Build model spec cards for the info display area."""
    if not selected_ids:
        return "*Select models to see specs.*"

    parts: list[str] = []
    for sid in selected_ids:
        model = next((m for m in _cached_models if m.id == sid), None)
        if model:
            parts.append(
                f"**{model.name}** · `{model.id}`\n"
                f"> Context: **{model.context_length:,}** · "
                f"Max output: **{model.max_completion:,}** · "
                f"Modality: `{model.modality}`\n"
                f"> {model.description}"
            )
    return "\n\n".join(parts)


def handle_preset_change(preset_name: str) -> tuple[str, float, float]:
    """Resolve preset prompt and smart defaults."""
    prompt = resolve_preset_prompt(preset_name)
    temp, top_p = resolve_smart_defaults(preset_name)
    return prompt, temp, top_p


def handle_cancel() -> dict:
    """Signal cancellation to the network module."""
    request_cancel()
    return gr.update(interactive=False, value="⏳ Cancelling...")


def handle_export(csv_data: str, json_data: str) -> tuple:
    """Write export files and return paths for download components."""
    csv_path, json_path = write_export_files(csv_data, json_data)
    return csv_path, json_path


def normalize_suite(val: str) -> str | None:
    """Convert '(none)' sentinel to None for the benchmark runner."""
    return val if val and val != "(none)" else None


# ── Output Index Constants ──────────────────────────────────────────────────
# Named indices for the 13-element output tuple from run_benchmark.

_OUT_LOG = 0
_OUT_INSIGHT = 1
_OUT_LEADERBOARD = 2
_OUT_BAR = 3
_OUT_SCATTER = 4
_OUT_CONSISTENCY = 5
_OUT_RADAR = 6
_OUT_SIDEBYSIDE = 7
_OUT_CSV = 8
_OUT_JSON = 9
_OUT_HISTORY = 10
_OUT_SHARE_MD = 11
_OUT_CANCEL_BTN = 12
_NUM_OUTPUTS = 13


def _empty_outputs(
    message: str,
    prompt_history: list[str],
) -> tuple:
    """Construct a full-width output tuple with placeholder values."""
    ef = empty_figure()
    out = [""] * _NUM_OUTPUTS
    out[_OUT_LOG] = message
    out[_OUT_LEADERBOARD] = pd.DataFrame()
    out[_OUT_BAR] = ef
    out[_OUT_SCATTER] = ef
    out[_OUT_CONSISTENCY] = ef
    out[_OUT_RADAR] = ef
    out[_OUT_HISTORY] = prompt_history
    out[_OUT_CANCEL_BTN] = gr.update(
        interactive=True, value="⛔ Cancel Benchmark"
    )
    return tuple(out)


# ── Main Benchmark Orchestrator ─────────────────────────────────────────────

def run_benchmark(
    selected_models: list[str],
    prompt_text: str,
    max_tokens: float,
    runs_per_model: float,
    parallel: bool,
    temperature: float,
    top_p: float,
    blind_mode: bool,
    suite_name: str | None,
    prompt_history: list[str],
    progress: gr.Progress = gr.Progress(track_tqdm=False),
):
    """
    Main benchmark orchestrator with suite support, blind mode,
    parallel/sequential dispatch, and cancellation.

    This is a Gradio generator function (yields intermediate progress).

    Yields:
        13-element tuples matching the output component list.

    Data flow per iteration:
        network.run_single_benchmark()
          → processing.aggregate_stats()
            → processing.build_leaderboard_rows()
              → visualization.build_*_chart()
              → insights.generate_insights()
              → export.export_results_*()
                → Gradio output components
    """
    reset_cancel()

    # ── Validation ──────────────────────────────────────────────────────
    if not get_api_key():
        yield _empty_outputs("⚠️ Set your API key first.", prompt_history)
        return

    if not selected_models:
        yield _empty_outputs("⚠️ Select at least one model.", prompt_history)
        return

    # ── Resolve prompts ─────────────────────────────────────────────────
    prompts_to_run: list[tuple[str, str]] = []

    if suite_name:
        prompts_to_run = resolve_suite_prompts(suite_name)

    if not prompts_to_run:
        if not prompt_text.strip():
            yield _empty_outputs(
                "⚠️ Enter a prompt or select a suite.", prompt_history
            )
            return
        prompts_to_run = [(prompt_text.strip(), "")]

    # ── Update prompt history ───────────────────────────────────────────
    new_prompts = [p for p, _ in prompts_to_run]
    prompt_history = update_prompt_history(prompt_history, new_prompts)

    # ── Build model lookup for blind mode reveal ────────────────────────
    model_lookup: dict[str, str] = {}
    for m in _cached_models:
        model_lookup[m.id] = m.name

    # ── Name resolver (blind or open) ───────────────────────────────────
    def name_for(model_id: str) -> str:
        if blind_mode:
            return apply_blind_labels(selected_models, model_id)
        return model_lookup.get(model_id, model_id)

    # ── Task list construction ──────────────────────────────────────────
    runs = int(runs_per_model)
    max_tok = int(max_tokens)

    task_args: list[tuple[str, str, str, int]] = []
    for prompt, label in prompts_to_run:
        for mid in selected_models:
            for run_idx in range(runs):
                task_args.append((mid, prompt, label, run_idx))

    total_tasks = len(task_args)
    completed = 0
    all_results: list[BenchmarkResult] = []
    log_lines: list[str] = []
    start_wall = time.perf_counter()

    # ── Log header ──────────────────────────────────────────────────────
    suite_tag = f" | Suite: **{suite_name}**" if suite_name else ""
    mode_tag = "🙈 Blind Mode" if blind_mode else "👁️ Open Mode"
    exec_tag = "⚡ Parallel" if parallel else "📶 Sequential"

    log_lines.append(
        f"### Benchmark Started — "
        f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC"
    )
    log_lines.append(
        f"Models: **{len(selected_models)}** · "
        f"Prompts: **{len(prompts_to_run)}** · "
        f"Runs/model/prompt: **{runs}** · "
        f"Max tokens: **{max_tok}** · "
        f"Temp: **{temperature}** · Top-p: **{top_p}** · "
        f"{mode_tag} · {exec_tag}{suite_tag}\n"
    )

    ef = empty_figure()

    # ── Progress yield helper ───────────────────────────────────────────
    def make_progress_yield() -> tuple:
        elapsed = time.perf_counter() - start_wall
        if completed > 0:
            eta = (elapsed / completed) * (total_tasks - completed)
            eta_str = f" · ETA: **{eta:.0f}s**"
        else:
            eta_str = ""

        status_line = f"*Progress: {completed}/{total_tasks}{eta_str}*"
        return (
            "\n".join(log_lines) + f"\n\n{status_line}",
            "",
            pd.DataFrame(),
            ef, ef, ef, ef,
            "",
            "", "",
            prompt_history,
            "",
            gr.update(interactive=True, value="⛔ Cancel Benchmark"),
        )

    # ── Single-task executor ────────────────────────────────────────────
    def execute_run(
        model_id: str,
        prompt: str,
        label: str,
        _run_idx: int,
    ) -> BenchmarkResult:
        return run_single_benchmark(
            model_id=model_id,
            model_name=name_for(model_id),
            prompt=prompt,
            max_tokens=max_tok,
            temperature=temperature,
            top_p=top_p,
            suite_label=label,
        )

    # ── Log a completed result ──────────────────────────────────────────
    def log_result(
        result: BenchmarkResult,
        label: str,
        run_idx: int,
    ) -> None:
        status = "✅" if not result.is_error else "❌"
        label_tag = f" [{label}]" if label else ""
        error_tag = (
            f" · ❌ {result.error[:50]}" if result.is_error else ""
        )
        log_lines.append(
            f"{status} `{result.model_name}` run {run_idx + 1}"
            f"{label_tag}: **{result.latency_sec}s** · "
            f"**{result.tokens_per_sec} tok/s**{error_tag}"
        )

    # ── Dispatch: parallel or sequential ────────────────────────────────
    if parallel:
        worker_count = min(MAX_PARALLEL_WORKERS, total_tasks)
        futures: dict[concurrent.futures.Future, tuple] = {}

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=worker_count,
        ) as pool:
            for args in task_args:
                if is_cancelled():
                    break
                future = pool.submit(execute_run, *args)
                futures[future] = args

            for future in concurrent.futures.as_completed(futures):
                if is_cancelled():
                    log_lines.append("\n### ⛔ Cancelled by user")
                    break

                args = futures[future]
                result = future.result()
                all_results.append(result)
                completed += 1

                log_result(result, args[2], args[3])
                progress(
                    completed / total_tasks,
                    desc=f"{completed}/{total_tasks}",
                )
                yield make_progress_yield()
    else:
        for args in task_args:
            if is_cancelled():
                log_lines.append("\n### ⛔ Cancelled by user")
                break

            result = execute_run(*args)
            all_results.append(result)
            completed += 1

            log_result(result, args[2], args[3])
            progress(
                completed / total_tasks,
                desc=f"{completed}/{total_tasks}",
            )
            yield make_progress_yield()

    # ── No results guard ────────────────────────────────────────────────
    if not all_results:
        yield _empty_outputs("No results collected.", prompt_history)
        return

    # ── Blind mode reveal ───────────────────────────────────────────────
    if blind_mode:
        reveal_blind_results(all_results, model_lookup)

    # ── Aggregation pipeline ────────────────────────────────────────────
    #   network results → processing → visualization/insights/export
    model_stats = aggregate_stats(all_results)
    rows = build_leaderboard_rows(model_stats)

    # Processing → Pandas DataFrame (PyArrow-backed)
    leaderboard_df = build_leaderboard_dataframe(rows)

    # Processing → Markdown
    sidebyside_md = build_sidebyside_markdown(model_stats)

    # Insights
    insight_text = generate_insights(rows)

    # Visualization (Plotly Figures)
    bar_fig = build_bar_chart(model_stats)
    scatter_fig = build_scatter_chart(model_stats)
    consistency_fig = build_consistency_chart(model_stats)
    radar_fig = build_radar_chart(rows)

    # Export
    csv_data = export_results_csv(all_results)
    json_data = export_results_json(all_results)
    share_md = build_share_markdown(rows, insight_text)

    # ── Final log entry ─────────────────────────────────────────────────
    elapsed_total = round(time.perf_counter() - start_wall, 1)
    cancelled_tag = " (partial — cancelled)" if is_cancelled() else ""
    log_lines.append(
        f"\n### ✅ Complete{cancelled_tag} — "
        f"{elapsed_total}s total — {len(all_results)} results"
    )

    # ── Yield final output tuple ────────────────────────────────────────
    yield (
        "\n".join(log_lines),       # 0:  log
        insight_text,                # 1:  insight
        leaderboard_df,              # 2:  leaderboard DataFrame
        bar_fig,                     # 3:  bar chart
        scatter_fig,                 # 4:  scatter chart
        consistency_fig,             # 5:  consistency chart
        radar_fig,                   # 6:  radar chart
        sidebyside_md,               # 7:  side-by-side responses
        csv_data,                    # 8:  csv state
        json_data,                   # 9:  json state
        prompt_history,              # 10: history state
        share_md,                    # 11: share markdown state
        gr.update(                   # 12: cancel button reset
            interactive=True,
            value="⛔ Cancel Benchmark",
        ),
    )


# ── Gradio Blocks Layout ───────────────────────────────────────────────────

def build_app() -> gr.Blocks:
    """
    Construct and return the complete Gradio Blocks application.

    The app object is returned (not launched) so that run.py can
    configure launch parameters independently.
    """
    with gr.Blocks(
        title="OpenRouter Free Model Benchmarker v3",
        css=CSS,
    ) as app:

        # ── Session State ───────────────────────────────────────────────
        prompt_history_state = gr.State([])
        csv_state = gr.State("")
        json_state = gr.State("")
        share_md_state = gr.State("")

        # ── Header ──────────────────────────────────────────────────────
        gr.Markdown(
            "# ⚡ OpenRouter Free Model Benchmarker v3\n"
            "**Discover → Filter → Configure → Benchmark → Analyze → Share**",
            elem_classes="header-block",
        )

        # ── API Key ────────────────────────────────────────────────────
        with gr.Accordion(
            "🔑 API Key", open=not bool(OPENROUTER_API_KEY)
        ):
            with gr.Row():
                api_key_input = gr.Textbox(
                    label="OpenRouter API Key",
                    type="password",
                    placeholder="sk-or-v1-...",
                    value=OPENROUTER_API_KEY,
                    info=(
                        "Free tier works — "
                        "https://openrouter.ai/settings/keys"
                    ),
                    scale=3,
                )
                key_status = gr.Textbox(
                    label="Status",
                    interactive=False,
                    scale=1,
                    value=(
                        "✅ Loaded from env"
                        if OPENROUTER_API_KEY
                        else "⚠️ No key"
                    ),
                )

            api_key_input.change(
                handle_set_api_key, api_key_input, key_status
            )

        # ── Model Discovery ─────────────────────────────────────────────
        gr.Markdown("---\n### 🔍 Model Discovery")

        with gr.Row():
            provider_filter = gr.Dropdown(
                label="Filter by Provider",
                choices=[],
                multiselect=True,
                info="e.g. google, meta-llama, mistralai",
                scale=1,
            )
            refresh_btn = gr.Button(
                "🔄 Discover Free Models",
                variant="primary",
                size="lg",
                scale=1,
            )
            model_status = gr.Markdown("*Click discover to fetch models*")

        model_selector = gr.Dropdown(
            label="Select Models to Benchmark",
            choices=[],
            multiselect=True,
            allow_custom_value=False,
            info="Multi-select — choose 2+ for comparison",
        )
        model_info_display = gr.Markdown("")

        refresh_btn.click(
            handle_refresh_models,
            outputs=[model_selector, provider_filter, model_status],
        )
        provider_filter.change(
            handle_filter_by_provider,
            provider_filter,
            model_selector,
        )
        model_selector.change(
            handle_model_info, model_selector, model_info_display
        )

        # ── Configuration ───────────────────────────────────────────────
        gr.Markdown("---\n### ⚙️ Configuration")

        with gr.Row():
            with gr.Column(scale=3):
                prompt_input = gr.Textbox(
                    label="Benchmark Prompt",
                    lines=4,
                    placeholder=(
                        "Enter your prompt, pick a preset, "
                        "or use a suite →"
                    ),
                )
                token_counter = gr.Markdown(
                    "~0 tokens", elem_classes="token-counter"
                )
            with gr.Column(scale=1):
                suite_dropdown = gr.Dropdown(
                    label="🧪 Benchmark Suite",
                    choices=["(none)"] + list(BENCHMARK_SUITES.keys()),
                    value="(none)",
                    info=(
                        "Overrides single prompt — "
                        "runs multiple presets"
                    ),
                )
                preset_dropdown = gr.Dropdown(
                    label="Single Preset",
                    choices=list(BENCHMARK_PRESETS.keys()),
                    value=None,
                )
                history_dropdown = gr.Dropdown(
                    label="Prompt History",
                    choices=[],
                    value=None,
                    info="Session history",
                )

        prompt_input.change(estimate_tokens, prompt_input, token_counter)
        history_dropdown.change(
            lambda x: x if x else "",
            history_dropdown,
            prompt_input,
        )

        with gr.Row():
            max_tokens_slider = gr.Slider(
                label="Max Tokens",
                minimum=32,
                maximum=4096,
                value=512,
                step=32,
            )
            runs_slider = gr.Slider(
                label="Runs / Model",
                minimum=1,
                maximum=10,
                value=1,
                step=1,
                info="↑ = better variance data",
            )

        with gr.Row():
            temperature_slider = gr.Slider(
                label="Temperature",
                minimum=0.0,
                maximum=2.0,
                value=0.7,
                step=0.05,
            )
            top_p_slider = gr.Slider(
                label="Top-p",
                minimum=0.0,
                maximum=1.0,
                value=1.0,
                step=0.05,
            )
            parallel_check = gr.Checkbox(
                label="⚡ Parallel", value=True
            )
            blind_mode_check = gr.Checkbox(
                label="🙈 Blind Mode",
                value=False,
                info="Hide names during run",
            )

        preset_dropdown.change(
            handle_preset_change,
            preset_dropdown,
            [prompt_input, temperature_slider, top_p_slider],
        )

        with gr.Row():
            run_btn = gr.Button(
                "🚀 Run Benchmark",
                variant="primary",
                size="lg",
                scale=3,
            )
            cancel_btn = gr.Button(
                "⛔ Cancel Benchmark",
                variant="stop",
                size="lg",
                scale=1,
            )

        # ── Results ─────────────────────────────────────────────────────
        gr.Markdown("---\n### 📊 Results")

        insight_output = gr.Markdown("", elem_classes="insight-box")

        with gr.Tabs():
            with gr.Tab("📋 Leaderboard"):
                leaderboard_df_output = gr.Dataframe(
                    label=(
                        "Sortable — click column headers to re-rank"
                    ),
                    interactive=False,
                    wrap=True,
                )

            with gr.Tab("📊 Charts"):
                bar_chart_output = gr.Plot(label="Metric Comparison")
                scatter_chart_output = gr.Plot(
                    label="TTFT vs Throughput"
                )

            with gr.Tab("🎯 Radar"):
                radar_chart_output = gr.Plot(
                    label="Normalized 5-Axis Comparison"
                )

            with gr.Tab("📈 Consistency"):
                consistency_chart_output = gr.Plot(
                    label="Variance (needs ≥2 runs/model)"
                )

            with gr.Tab("🔀 Side-by-Side"):
                sidebyside_output = gr.Markdown(
                    "*Run with 2+ models to compare outputs.*"
                )

            with gr.Tab("📝 Live Log"):
                log_output = gr.Markdown("*Waiting for benchmark...*")

            with gr.Tab("💾 Export & Share"):
                gr.Markdown(
                    "**Download** raw data or "
                    "**copy** a share-ready summary:"
                )
                with gr.Row():
                    csv_download = gr.File(
                        label="CSV", interactive=False
                    )
                    json_download = gr.File(
                        label="JSON", interactive=False
                    )
                export_btn = gr.Button(
                    "📥 Generate Export Files",
                    variant="secondary",
                )
                gr.Markdown("---")
                share_md_output = gr.Code(
                    label=(
                        "📋 Share-Ready Markdown "
                        "(select all + copy for Reddit/Discord)"
                    ),
                    language="markdown",
                    lines=12,
                    interactive=False,
                )

        with gr.Accordion("📖 Metric Glossary", open=False):
            gr.Markdown(METRIC_GLOSSARY, elem_classes="metric-glossary")

        # ── Event Wiring ────────────────────────────────────────────────

        export_btn.click(
            handle_export,
            [csv_state, json_state],
            [csv_download, json_download],
        )

        cancel_btn.click(handle_cancel, outputs=[cancel_btn])

        run_btn.click(
            normalize_suite,
            inputs=[suite_dropdown],
            outputs=[suite_dropdown],
        ).then(
            run_benchmark,
            inputs=[
                model_selector,
                prompt_input,
                max_tokens_slider,
                runs_slider,
                parallel_check,
                temperature_slider,
                top_p_slider,
                blind_mode_check,
                suite_dropdown,
                prompt_history_state,
            ],
            outputs=[
                log_output,
                insight_output,
                leaderboard_df_output,
                bar_chart_output,
                scatter_chart_output,
                consistency_chart_output,
                radar_chart_output,
                sidebyside_output,
                csv_state,
                json_state,
                prompt_history_state,
                share_md_state,
                cancel_btn,
            ],
        ).then(
            lambda history: gr.update(
                choices=history_to_choices(history),
                value=None,
            ),
            inputs=[prompt_history_state],
            outputs=[history_dropdown],
        ).then(
            lambda md: md,
            inputs=[share_md_state],
            outputs=[share_md_output],
        ).then(
            lambda: gr.update(
                interactive=True, value="⛔ Cancel Benchmark"
            ),
            outputs=[cancel_btn],
        )

        # ── Footer ─────────────────────────────────────────────────────
        gr.Markdown(
            "---\n<center>"
            "v3 · Radar · Blind Mode · Suites · Cancel · Smart Defaults · "
            "<a href='https://openrouter.ai' target='_blank'>OpenRouter</a>"
            "</center>",
        )

    return app
