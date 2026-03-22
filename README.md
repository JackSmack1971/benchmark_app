# ⚡ OpenRouter Free Model Benchmarker v3

Head-to-head benchmarking of free LLM models via [OpenRouter](https://openrouter.ai), built with Gradio.

## Architecture

```
benchmark_app/
├── __init__.py          # Package metadata
├── config.py            # Constants, presets, suites, dataclasses
├── network.py           # httpx.AsyncClient + async fetching & streaming
├── processing.py        # PyArrow-backed Pandas aggregation, statistics
├── visualization.py     # Plotly chart builders (bar, scatter, consistency, radar)
├── export.py            # CSV, JSON, share-ready Markdown
├── insights.py          # Natural-language insight generation
├── ui_components.py     # Pure Gradio UI declarations and blocks
├── state_managers.py    # Async orchestrator and business logic binding
├── app.py               # Thin entry point wrapper
├── run.py               # Entry point
├── requirements.txt     # Pinned dependencies
├── tests/               # Comprehensive pytest suite (async + pyarrow coverage)
├── .env.example         # Environment variable template
└── .gitignore           # Git ignore rules
```

**Data flow (unidirectional):**

```
network.py → processing.py (PyArrow) → visualization.py (Plotly) → state_managers.py → ui_components.py
```

**Module isolation:**

| Module               | gradio | httpx/asyncio | pandas | plotly | config |
|----------------------|:------:|:-------------:|:------:|:------:|:------:|
| `config.py`          |   ✗    |       ✗       |   ✗    |   ✗    |   —    |
| `network.py`         |   ✗    |       ✅      |   ✗    |   ✗    |   ✅   |
| `processing.py`      |   ✗    |       ✗       |   ✅   |   ✗    |   ✅   |
| `visualization.py`   |   ✗    |       ✗       |   ✗    |   ✅   |   ✅   |
| `export.py`          |   ✗    |       ✗       |   ✗    |   ✗    |   ✅   |
| `insights.py`        |   ✗    |       ✗       |   ✗    |   ✗    |   ✅   |
| `state_managers.py`  |   ✅   |       ✅      |   ✅   |   ✗    |   ✅   |
| `ui_components.py`   |   ✅   |       ✗       |   ✗    |   ✗    |   ✅   |
| `app.py` / `run.py`  |   ✅   |       ✗       |   ✗    |   ✗    |   ✗    |

## Quick Start

```bash
# Clone and enter the directory
cd benchmark_app

# Create virtual environment
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt

# Set your API key (free tier works)
cp .env.example .env
# Edit .env and add your key from https://openrouter.ai/settings/keys

# Launch
python run.py
```

Open `http://localhost:7860` in your browser.

## Features

- **Model Discovery** — Auto-fetch all free text models from OpenRouter
- **Provider Filtering** — Filter by google, meta-llama, mistralai, etc.
- **Benchmark Presets** — Quick Reasoning, Creative Writing, Code Gen, Math, and more
- **Benchmark Suites** — Run multiple presets and average results
- **Blind Mode** — Hide model names during runs, reveal after voting
- **Parallel Execution** — ThreadPoolExecutor with configurable worker count
- **Cancellation** — Abort mid-benchmark via threading.Event
- **Smart Defaults** — Auto-set temperature/top_p based on prompt category
- **Live Progress** — ETA estimation and per-run status updates
- **5 Visualization Tabs:**
  - Sortable leaderboard (PyArrow-backed DataFrame)
  - Bar charts (latency, TTFT, throughput)
  - Bubble scatter (TTFT vs throughput vs output size)
  - Box plots (consistency across multi-run benchmarks)
  - 5-axis radar (Speed, Responsiveness, Consistency, Output Volume, Reliability)
- **Side-by-Side** — Compare model responses directly
- **Insight Engine** — Auto-generated natural-language analysis
- **Export** — CSV, JSON, and Reddit/Discord-ready Markdown

## Environment Variables

| Variable | Required | Default | Description |
|----------|:--------:|---------|-------------|
| `OPENROUTER_API_KEY` | Yes | — | Your OpenRouter API key |
| `GRADIO_SERVER_NAME` | No | `0.0.0.0` | Server bind address |
| `GRADIO_SERVER_PORT` | No | `7860` | Server port |
| `GRADIO_SHARE` | No | `false` | Enable Gradio public share link |

## Key Design Decisions

**Persistent `httpx.AsyncClient`** for high-performance async socket multiplexing, coupled with native asyncio exponential backoff for transient errors (429, 5xx).

**Async concurrency model** via `asyncio.gather/as_completed` — eliminates thread blocking during concurrent streaming benchmark execution, avoiding race conditions.

**PyArrow dtype backend** on all Pandas DataFrames for columnar memory layout, native nullable types (no NaN sentinel), and reduced memory footprint.

**Typed dataclasses** (`BenchmarkResult`, `ModelInfo`, `LeaderboardRow`) replace anonymous dicts throughout the pipeline — static analysis catches typos at write time, not runtime.

**Separation of Concerns** — The legacy `app.py` God Class was decoupled. `ui_components.py` is purely functional layout, while `state_managers.py` orchestrates state mutations. Neither contains explicit HTTP streaming or statistics computations. Every domain concern is delegated to its respective module.

## License

MIT
