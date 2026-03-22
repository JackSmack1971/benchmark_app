"""
config.py — Central Configuration Module
─────────────────────────────────────────
All constants, presets, benchmark suites, smart defaults, chart colors,
metric glossary, and dataclass definitions. Zero business logic.

Every other module imports from here; this module imports from nothing
internal (stdlib + dataclass only).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Optional


# ── Environment & API ───────────────────────────────────────────────────────

OPENROUTER_BASE: str = "https://openrouter.ai/api/v1"

HEADERS_BASE: dict[str, str] = {
    "Content-Type": "application/json",
    "HTTP-Referer": "http://localhost:7860",
    "X-Title": "Free Model Benchmarker v3",
}


# ── Network Tuning ──────────────────────────────────────────────────────────

SESSION_CONNECT_TIMEOUT: float = 5.0
SESSION_READ_TIMEOUT: float = 30.0
STREAM_READ_TIMEOUT: float = 120.0
RETRY_TOTAL: int = 3
RETRY_BACKOFF_FACTOR: float = 0.5
RETRY_STATUS_FORCELIST: tuple[int, ...] = (429, 500, 502, 503, 504)
MAX_PARALLEL_WORKERS: int = 8


# ── Benchmark Presets ───────────────────────────────────────────────────────

BENCHMARK_PRESETS: dict[str, dict[str, str]] = {
    "⚡ Quick Reasoning": {
        "prompt": (
            "Explain why 0.999... repeating equals 1. "
            "Be concise but rigorous."
        ),
        "category": "reasoning",
    },
    "🎨 Creative Writing": {
        "prompt": (
            "Write a 150-word flash fiction piece about a mass extinction "
            "event told from the perspective of a single bacterium."
        ),
        "category": "creative",
    },
    "💻 Code Generation": {
        "prompt": (
            "Write a Python function that finds the longest palindromic "
            "substring in a given string. Include type hints, docstring, "
            "and O(n²) time complexity. Return the function only."
        ),
        "category": "code",
    },
    "📝 Summarization": {
        "prompt": (
            "Summarize the concept of quantum entanglement in exactly "
            "3 sentences suitable for a college freshman."
        ),
        "category": "reasoning",
    },
    "🎯 Instruction Following": {
        "prompt": (
            "List exactly 5 countries whose names start with the letter 'M'. "
            "Format as a numbered list. Do not include any additional text."
        ),
        "category": "instruction",
    },
    "🧠 Chain of Thought": {
        "prompt": (
            "A bat and a ball cost $1.10 in total. The bat costs $1.00 more "
            "than the ball. How much does the ball cost? Show your reasoning "
            "step by step."
        ),
        "category": "reasoning",
    },
    "🔬 Technical Analysis": {
        "prompt": (
            "Explain the difference between TCP and UDP. Include: when to "
            "use each, header size, connection overhead, and one real-world "
            "use case per protocol."
        ),
        "category": "reasoning",
    },
    "🔢 Math / Logic": {
        "prompt": (
            "Find all positive integers n such that n² + n + 41 is a "
            "perfect square. Show your work."
        ),
        "category": "math",
    },
}


# ── Smart Defaults (per category) ──────────────────────────────────────────

SMART_DEFAULTS: dict[str, dict[str, float]] = {
    "reasoning":    {"temperature": 0.3, "top_p": 0.9},
    "creative":     {"temperature": 0.9, "top_p": 0.95},
    "code":         {"temperature": 0.1, "top_p": 0.95},
    "math":         {"temperature": 0.0, "top_p": 1.0},
    "instruction":  {"temperature": 0.0, "top_p": 1.0},
}

DEFAULT_TEMPERATURE: float = 0.7
DEFAULT_TOP_P: float = 1.0


# ── Benchmark Suites ────────────────────────────────────────────────────────

BENCHMARK_SUITES: dict[str, list[str]] = {
    "🧪 Full Reasoning Suite": [
        "⚡ Quick Reasoning",
        "🧠 Chain of Thought",
        "🔢 Math / Logic",
    ],
    "💼 Practical Suite": [
        "💻 Code Generation",
        "📝 Summarization",
        "🎯 Instruction Following",
    ],
    "🌈 All Presets": list(BENCHMARK_PRESETS.keys()),
}


# ── Chart Palette ───────────────────────────────────────────────────────────

CHART_COLORS: list[str] = [
    "#3b82f6", "#ef4444", "#22c55e", "#f59e0b", "#8b5cf6",
    "#06b6d4", "#ec4899", "#14b8a6", "#f97316", "#6366f1",
    "#84cc16", "#e11d48", "#0ea5e9", "#a855f7", "#10b981",
]


# ── Metric Glossary (Markdown) ──────────────────────────────────────────────

METRIC_GLOSSARY: str = """
| Metric | Definition |
|:---|:---|
| **Latency** | Total wall-clock time from request to final token (includes network RTT) |
| **TTFT** | Time To First Token — seconds until the first SSE content chunk arrives |
| **tok/s** | Throughput — completion tokens per second of total latency |
| **Completion Tokens** | Tokens in the model's response (API-reported or estimated) |
| **σ (Std Dev)** | Standard deviation across runs — lower = more consistent |
| **CV%** | Coefficient of Variation (σ/mean × 100) — normalized consistency |
| **Radar axes** | Speed (1/latency), Responsiveness (1/TTFT), Consistency (1/CV%), Output Volume, Reliability (1 − error rate) — all normalized 0–100 |
| **Cost/1K tokens** | Estimated USD cost per 1 000 completion tokens based on OpenRouter list pricing (prompt + completion costs combined) |
"""


# ── UI Constants ────────────────────────────────────────────────────────────

CSS: str = """
.header-block { text-align: center; padding: 0.5rem 0 0.2rem; }
footer { display: none !important; }
.metric-glossary { font-size: 0.85em; opacity: 0.85; }
.token-counter { font-size: 0.8em; opacity: 0.65; font-family: 'JetBrains Mono', monospace; }
.insight-box { border-left: 3px solid #3b82f6; padding-left: 1rem; }
"""

MAX_PROMPT_HISTORY: int = 20
PROMPT_PREVIEW_LEN: int = 60


# ── Data Structures ─────────────────────────────────────────────────────────

@dataclass
class BenchmarkResult:
    """Immutable record of a single benchmark run against one model."""

    model_id: str
    model_name: str
    prompt: str
    response: str
    latency_sec: float
    ttft_sec: Optional[float]
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    tokens_per_sec: float
    temperature: float = 0.7
    top_p: float = 1.0
    max_tokens_cfg: int = 512
    error: Optional[str] = None
    timestamp: str = ""
    suite_label: str = ""
    prompt_cost_usd: float = 0.0
    completion_cost_usd: float = 0.0

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict:
        """Serialization helper — avoids importing dataclasses.asdict elsewhere."""
        return asdict(self)

    @property
    def is_error(self) -> bool:
        return self.error is not None


@dataclass
class ModelInfo:
    """Parsed free-model metadata from OpenRouter /models endpoint."""

    id: str
    name: str
    provider: str
    context_length: int
    modality: str
    max_completion: int
    description: str
    prompt_price: float = 0.0
    completion_price: float = 0.0

    @classmethod
    def from_api_dict(cls, m: dict) -> "ModelInfo":
        """Factory: construct from a raw OpenRouter model dict."""
        model_id = m.get("id", "unknown")
        provider = model_id.split("/")[0] if "/" in model_id else "unknown"
        pricing = m.get("pricing", {})
        try:
            prompt_price = float(pricing.get("prompt", 0) or 0)
            completion_price = float(pricing.get("completion", 0) or 0)
        except (ValueError, TypeError):
            prompt_price = 0.0
            completion_price = 0.0
        return cls(
            id=model_id,
            name=m.get("name", model_id),
            provider=provider,
            context_length=m.get("context_length", 0),
            modality=m.get("architecture", {}).get("modality", ""),
            max_completion=m.get("top_provider", {}).get(
                "max_completion_tokens", 0
            ),
            description=(m.get("description") or "")[:200],
            prompt_price=prompt_price,
            completion_price=completion_price,
        )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class LeaderboardRow:
    """One row in the aggregated leaderboard. Typed for downstream consumers."""

    model: str
    model_id: str
    avg_lat: float
    std_lat: float
    avg_ttft: Optional[float]
    avg_tps: float
    std_tps: float
    cv_tps: float
    avg_tokens: int
    errors: int
    runs: int
    error_rate: float
    avg_cost_per_1k: float = 0.0
    composite_score: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)
