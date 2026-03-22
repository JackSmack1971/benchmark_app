#!/usr/bin/env python3
"""
run.py — Application Entry Point
─────────────────────────────────
Bootstraps the benchmark application:
  1. Loads environment variables from .env (if present)
  2. Builds the Gradio Blocks app via app.build_app()
  3. Launches with configurable server parameters and theme

Usage:
    python run.py

Environment variables:
    OPENROUTER_API_KEY  — OpenRouter API key (required for benchmarking)
    GRADIO_SERVER_NAME  — Bind address (default: 0.0.0.0)
    GRADIO_SERVER_PORT  — Port number (default: 7860)
    GRADIO_SHARE        — Set to "true" for public Gradio link
"""

from __future__ import annotations

import os
import sys

# ── Optional .env loading ───────────────────────────────────────────────────
# python-dotenv is a soft dependency — if not installed, .env is skipped.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import gradio as gr

from app import build_app
from config import CSS


def main() -> None:
    """
    Build and launch the Gradio application.

    All launch parameters are configurable via environment variables
    with sensible defaults for local development.
    """
    # ── Parse environment configuration ─────────────────────────────────
    server_name: str = os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0")
    server_port: int = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
    share: bool = os.environ.get("GRADIO_SHARE", "false").lower() == "true"

    # ── Validate API key presence ───────────────────────────────────────
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        print(
            "⚠️  OPENROUTER_API_KEY not found in environment.\n"
            "   You can set it in the UI, via .env file, or export it:\n"
            "   export OPENROUTER_API_KEY=sk-or-v1-...\n",
            file=sys.stderr,
        )

    # ── Build application ───────────────────────────────────────────────
    application = build_app()

    # ── Theme ───────────────────────────────────────────────────────────
    theme = gr.themes.Base(
        primary_hue="zinc",
        secondary_hue="sky",
        neutral_hue="zinc",
        font=gr.themes.GoogleFont("JetBrains Mono"),
    )

    # ── Launch ──────────────────────────────────────────────────────────
    print(
        f"\n🚀 OpenRouter Free Model Benchmarker v3\n"
        f"   Server:  http://{server_name}:{server_port}\n"
        f"   Share:   {'enabled' if share else 'disabled'}\n"
        f"   API Key: {'✅ loaded' if api_key else '⚠️  not set'}\n"
    )

    application.launch(
        server_name=server_name,
        server_port=server_port,
        share=share,
        theme=theme,
        css=CSS,
    )


if __name__ == "__main__":
    main()
