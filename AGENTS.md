# AGENTS.md — OpenRouter Free Model Benchmarker v3

## Project Overview

Single-package Python app (Gradio 6.9+) for head-to-head benchmarking of free LLM models via the OpenRouter API. Strictly modular architecture with **unidirectional data flow**:

```
network.py → processing.py → visualization.py → app.py (Gradio shell)
```

`app.py` is a **thin routing/presentation layer only** — it owns zero business logic. Every domain concern (HTTP, stats, charts, export, insights) lives in its dedicated module. See the module isolation matrix in `README.md`.

## Project Layout

| File | Responsibility | Key Imports |
|---|---|---|
| `config.py` | Constants, presets, suites, dataclasses | stdlib only |
| `network.py` | `requests.Session`, model fetch, streaming benchmarks | `config`, `requests` |
| `processing.py` | PyArrow-backed Pandas aggregation, statistics | `config`, `pandas` |
| `visualization.py` | Plotly chart builders (bar, scatter, box, radar) | `config`, `plotly` |
| `export.py` | CSV, JSON, Markdown serialization | `config`, `csv/json` |
| `insights.py` | Natural-language insight generation | `config` only |
| `app.py` | Gradio Blocks UI, event wiring, orchestration | `config`, `network`, `processing`, `visualization`, `export`, `insights`, `gradio` |
| `run.py` | Entry point — env loading, theme, launch | `app`, `gradio` |

## Build & Run Commands

```bash
# Create venv and install
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Set API key (free tier works)
export OPENROUTER_API_KEY=sk-or-v1-...

# Launch app
python run.py
# → http://localhost:7860

# Run a single module check (no test suite exists yet)
python -c "from config import BenchmarkResult; print('config OK')"
python -c "from processing import aggregate_stats; print('processing OK')"
python -c "from visualization import empty_figure; print('visualization OK')"
```

**No test suite currently exists.** If adding tests, use `pytest` and place them in a `tests/` directory mirroring module names (e.g., `tests/test_processing.py`).

## Code Style & Conventions

### Do

- **Typed dataclasses** for all data structures (`BenchmarkResult`, `ModelInfo`, `LeaderboardRow` in `config.py`) — never anonymous dicts through the pipeline.
- **PyArrow dtype backend** on all Pandas DataFrames. Explicit `astype("float64[pyarrow]")` coercion. Use `pd.options.mode.copy_on_write = True`.
- **`statistics` module** for mean/stdev (numerically stable) — not manual math.
- **Reuse `requests.Session()`** with `HTTPAdapter` + `urllib3.util.Retry` — never bare `requests.get()`.
- **Thread-safe state** via `threading.Lock` (API key) and `threading.Event` (cancellation).
- **Explicit timeouts** on every HTTP call: `timeout=(connect, read)`.
- **Method chaining** in Pandas; vectorized ops only — no `.iterrows()`.
- **Return native Plotly `Figure` objects** directly to `gr.Plot`.
- **Docstrings** on every public function — module-level docstring on every file.
- **Type hints** on all function signatures (`from __future__ import annotations`).

### Don't

- Put business logic in `app.py` — it delegates everything.
- Use `global` for user-facing state — use `gr.State()` for session data.
- Use `inplace=True` on Pandas operations.
- Use bare `except:` — catch specific `requests.exceptions.*` subclasses.
- Import across module boundaries that violate the isolation matrix (e.g., `network.py` must never import `pandas` or `plotly`).
- Hardcode API keys or secrets — load from env vars / `.env`.
- Push raw unprocessed data to Plotly — aggregate in `processing.py` first.

### Naming

- Module-level private helpers: `_prefixed_with_underscore`
- Constants: `UPPER_SNAKE_CASE` in `config.py`
- Dataclass fields: `lower_snake_case`
- Gradio components: descriptive suffix (`_slider`, `_btn`, `_output`, `_state`)

## Architecture Rules

1. **Module isolation is enforced** — check the matrix in `README.md` before adding imports. If `visualization.py` needs data, it must come pre-computed from `processing.py`.
2. **`config.py` imports from nothing internal** — it is the dependency root.
3. **New benchmark metrics** → add field to `BenchmarkResult` or `LeaderboardRow` in `config.py`, then propagate through `processing.py` → `visualization.py` → `app.py`.
4. **New chart type** → add builder function in `visualization.py`, wire output in `app.py`, add to the `_NUM_OUTPUTS` tuple.
5. **New preset/suite** → add to `BENCHMARK_PRESETS` / `BENCHMARK_SUITES` / `SMART_DEFAULTS` in `config.py`. No other file changes needed.
6. **The 13-element output tuple** in `app.py` (`run_benchmark` generator) must stay synchronized with the `outputs=` list in the `.then()` chain. Use the `_OUT_*` index constants.

## Environment Variables

| Variable | Required | Default |
|---|:---:|---|
| `OPENROUTER_API_KEY` | Yes | — |
| `GRADIO_SERVER_NAME` | No | `0.0.0.0` |
| `GRADIO_SERVER_PORT` | No | `7860` |
| `GRADIO_SHARE` | No | `false` |

## Git Workflow

- **Branch naming**: `feature/<slug>`, `fix/<slug>`, `refactor/<slug>`
- **Commit format**: `feat(module): description` / `fix(module): description`
- **Pre-commit**: verify module isolation (no cross-boundary imports), run import smoke tests above
- **PR checklist**: passes smoke tests, docstrings on new functions, type hints present, no secrets in diff

## Gotchas & Pitfalls

- **Output tuple sync**: Adding/removing an output from `run_benchmark` requires updating `_NUM_OUTPUTS`, `_empty_outputs`, `make_progress_yield`, the final yield, AND the `.then()` `outputs=` list in `build_app()`. Miss one and Gradio silently breaks.
- **Blind mode mutation**: `reveal_blind_results()` mutates `BenchmarkResult.model_name` in-place — this is intentional but must happen after all results are collected and before aggregation.
- **Token estimation fallback**: When the API omits `usage`, `network.py` estimates tokens via word count heuristic (`words * 4/3`). This is approximate.
- **`_cached_models` is module-level** in `app.py` — not shared across Gradio sessions in 6.9+ (queue isolation), but be aware if refactoring.
- **No test suite yet** — if creating one, mock `requests.Session` in `network.py` tests and use fixture-based `BenchmarkResult` factories.

## When Stuck

1. Re-read the module-level docstring of the file you're editing — it states responsibilities and import constraints.
2. Check `config.py` for the canonical data structures before creating new dicts.
3. Trace data flow: `network` → `processing` → `visualization` → `app`. If data isn't where you expect, the break is at a module boundary.
4. If uncertain about a Gradio API, check `Best_Practices` in the repo root.
5. **Ask a clarifying question or propose a plan** rather than guessing at architectural decisions.
