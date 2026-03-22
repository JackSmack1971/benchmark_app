"""
ui_components.py — Gradio UI Declarations
─────────────────────────────────────────
Pure declarations for the user interface layout.
Wires UI signals to the handlers in state_managers.py.
"""

import os
import gradio as gr

from config import (
    BENCHMARK_PRESETS,
    BENCHMARK_SUITES,
    METRIC_GLOSSARY,
)
from processing import (
    estimate_tokens,
    history_to_choices,
)
from state_managers import (
    handle_key_status_update,
    handle_refresh_models,
    handle_filter_by_provider,
    handle_model_info,
    handle_preset_change,
    handle_cancel,
    handle_export,
    normalize_suite,
    run_benchmark,
)

def build_app() -> gr.Blocks:
    default_api_key = os.environ.get("OPENROUTER_API_KEY", "")

    with gr.Blocks(title="OpenRouter Free Model Benchmarker v3") as app:
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
