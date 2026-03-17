# ⚡ OpenRouter Free Model Benchmarker v3

Head-to-head benchmarking of free LLM models via [OpenRouter](https://openrouter.ai), built with Gradio.

## Quick Start

```bash
# Enter the directory
cd benchmark_app

# Create virtual environment
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt

# Set your API key (free tier works)
export OPENROUTER_API_KEY=sk-or-v1-...
# Or create a .env file: echo "OPENROUTER_API_KEY=sk-or-v1-..." > .env

# Launch
python run.py
```

Open `http://localhost:7860` in your browser.

## Features

### Model Discovery & Filtering
- **Auto-discover free models** — one click fetches every free text model from OpenRouter and shows a count by provider
- **Provider filter** — narrow down to google, meta-llama, mistralai, and others via a multi-select dropdown
- **Model info** — select any model to see its context length, max output tokens, and modality before running

### Benchmark Configuration
- **Custom prompt** — type any prompt directly; a live token counter shows an estimated cost
- **8 built-in presets** — Quick Reasoning, Creative Writing, Code Generation, Summarization, Instruction Following, Chain of Thought, Technical Analysis, Math/Logic
- **3 benchmark suites** — Full Reasoning Suite, Practical Suite, and All Presets; suites run multiple prompts and aggregate results across all of them
- **Smart defaults** — selecting a preset automatically sets the recommended temperature and top-p for that category
- **Prompt history** — previously used prompts are saved per session and available as a dropdown for quick re-use
- **Runs per model** — run each model 1–10 times to collect variance and consistency data
- **Temperature** (0.0–2.0) and **Top-p** (0.0–1.0) sliders with step controls
- **Max tokens** — 32–4096, step 32
- **Parallel execution** — runs all selected models concurrently via async tasks (toggle off for sequential)
- **Blind mode** — hides model names as "Model #1", "Model #2", etc. during execution; names are revealed after the run completes

### Benchmark Execution
- Live progress bar with ETA estimation and per-run status in the log
- Cancel button aborts all in-flight requests immediately

### Results — 6 Tabs

| Tab | What you see |
|-----|-------------|
| **📋 Leaderboard** | Sortable table: rank (🥇🥈🥉), model name, avg latency, σ latency, avg TTFT, avg tok/s, σ tok/s, CV%, avg tokens, error count, run count |
| **📊 Charts** | Three-panel bar chart (latency, TTFT, throughput) and a WebGL-accelerated scatter plot (TTFT vs throughput, bubble size = token count) |
| **🎯 Radar** | Five-axis normalized radar chart: Speed, Responsiveness, Consistency, Output Volume, Reliability |
| **📈 Consistency** | Box plots showing latency and throughput distributions across runs (requires ≥ 2 runs per model) |
| **🔀 Side-by-Side** | Direct response comparison across models (up to 4 pairs, truncated at 1 200 characters each) |
| **📝 Live Log** | Timestamped per-run log with latency, tok/s, and any error messages |

### Insight Engine
After every run an auto-generated summary identifies the fastest throughput model, lowest-latency model (if different), best TTFT, most consistent model (multi-run only), output volume spread, and any models that produced errors.

### Export & Sharing
- **CSV** — 16-field export (timestamp, model IDs, all metrics, prompt, response)
- **JSON** — pretty-printed array of all result objects
- **Share-ready Markdown** — ranked table with medals and the insight summary, formatted for Reddit and Discord
- Files are written to `/tmp/benchmark_results.{csv,json}` and offered as download links

## Configuration

### Environment Variables

| Variable | Required | Default | Description |
|----------|:--------:|---------|-------------|
| `OPENROUTER_API_KEY` | Yes | — | Your OpenRouter API key ([get one free](https://openrouter.ai/settings/keys)) |
| `GRADIO_SERVER_NAME` | No | `0.0.0.0` | Server bind address |
| `GRADIO_SERVER_PORT` | No | `7860` | Server port |
| `GRADIO_SHARE` | No | `false` | Set to `true` to generate a public Gradio share link |

The API key can also be entered directly in the UI's **🔑 API Key** accordion without restarting.

### Metric Glossary

| Metric | Definition |
|:-------|:-----------|
| **Latency** | Total wall-clock time from request to final token |
| **TTFT** | Time To First Token — seconds until the first content chunk arrives |
| **tok/s** | Throughput — completion tokens per second of total latency |
| **σ (Std Dev)** | Standard deviation across runs |
| **CV%** | Coefficient of Variation (σ / mean × 100) — normalized consistency |
| **Radar axes** | Speed (1/latency), Responsiveness (1/TTFT), Consistency (1/CV%), Output Volume, Reliability (1 − error rate) — all normalized 0–100 |

## Architecture

```
benchmark_app/
├── config.py            # Constants, presets, suites, dataclasses
├── network.py           # httpx.AsyncClient, model fetch, streaming
├── processing.py        # PyArrow-backed Pandas aggregation, statistics
├── visualization.py     # Plotly chart builders (bar, scatter, consistency, radar)
├── export.py            # CSV, JSON, share-ready Markdown
├── insights.py          # Natural-language insight generation
├── app.py               # Thin Gradio UI — routing & presentation only
├── run.py               # Entry point
└── requirements.txt     # Pinned dependencies
```

**Data flow (unidirectional):**

```
network.py → processing.py (PyArrow) → visualization.py (Plotly) → app.py (Gradio)
```

**Module isolation matrix:**

| Module | gradio | httpx | pandas | plotly | config |
|---|:---:|:---:|:---:|:---:|:---:|
| `config.py` | ✗ | ✗ | ✗ | ✗ | — |
| `network.py` | ✗ | ✅ | ✗ | ✗ | ✅ |
| `processing.py` | ✗ | ✗ | ✅ | ✗ | ✅ |
| `visualization.py` | ✗ | ✗ | ✗ | ✅ | ✅ |
| `export.py` | ✗ | ✗ | ✗ | ✗ | ✅ |
| `insights.py` | ✗ | ✗ | ✗ | ✗ | ✅ |
| `app.py` | ✅ | ✗ | ✅ | ✗ | ✅ |
| `run.py` | ✅ | ✗ | ✗ | ✗ | ✗ |

## License

MIT
