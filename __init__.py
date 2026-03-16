"""
OpenRouter Free Model Benchmarker v3
─────────────────────────────────────
Modular Gradio application for head-to-head benchmarking of free LLM models.

Package structure:
    config.py         — Constants, presets, dataclasses
    network.py        — HTTP I/O, model discovery, streaming benchmarks
    processing.py     — PyArrow-backed Pandas aggregation, statistics
    visualization.py  — Plotly chart builders
    export.py         — CSV, JSON, Markdown serialization
    insights.py       — Natural-language insight engine
    app.py            — Thin Gradio UI shell
    run.py            — Application entry point
"""

__version__ = "3.0.0"
__author__ = "Bret"
