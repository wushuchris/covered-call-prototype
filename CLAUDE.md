# CLAUDE.md — Covered Call Prototype

## Project Overview

ML-Driven Covered Call Optimization for AAI-590 Capstone (USD / Validex Growth Investors).
Predicts which moneyness bucket (ATM, OTM5, OTM10) yields the best covered call return each month for 10 large-cap U.S. stocks.

**Ship date**: April 4, 2026.
**Branch**: `carlos` (infrastructure & utilities).
**Teammates**: Christopher (repo/data), Swathi (models), Fatimat (models/features).

---

## Architecture

Two-microservice system, both Python. No JS. No databases.

```
main.py                          — dev launcher (Popen, both services)
deploy/setup_services.sh         — systemd (gunicorn + uvicorn), auto-start

src/
├── utils.py                     — logger, log_call decorator, ServiceRequest (shared Pydantic model)
├── ui/
│   ├── app.py                   — FastHTML :8008 (routes, USD branding, local fonts)
│   ├── ui_components.py         — MonsterUI component tree (launcher, trading, docs screens)
│   ├── ui_handler.py            — aiohttp bridge to inference service
│   ├── ui_utils.py              — pack_request, send_to_inference
│   └── static/                  — logo, fonts (served locally, no external API calls)
└── inference/
    ├── app.py                   — FastAPI :8009 (/inference, /backtest endpoints)
    ├── model.py                 — model loading, feature store, predict_bucket()
    ├── daily.py                 — single-day inference orchestration
    ├── chart.py                 — OHLC candlestick data builder
    ├── strategy.py              — CoveredCallStrategy (NautilusTrader) + inference wrappers
    ├── backtesting.py           — NautilusTrader BacktestEngine + caching
    └── inference_utils.py       — input validation
```

### Data Flow

```
User sidebar → POST /inference_call → ui_handler.pack_request()
    → aiohttp POST to :8009/inference (ServiceRequest body)
    → daily.run_daily_inference() → strategy.simulate_inference() → model.predict_bucket()
    → JSON response → ui_handler → inference_results_card() → htmx swap
```

### Inter-Service Communication

Services communicate via `ServiceRequest` (Pydantic BaseModel in `src/utils.py`).
Zero-trust: neither service knows the other's internals, they only share the data format.

---

## Critical Rules

These are non-negotiable. Taken from `persistence_agent/microservices_grounding.md`.

1. **Python only** — No JavaScript. FastHTML + MonsterUI handle all UI. ApexCharts via `<uk-chart>`.
2. **No databases** — Minimal side-effects/persistence. Parquet files and JSON caches only.
3. **Async first** — Don't block the event loop. Use `async def` for all service entry points and inter-service calls.
4. **One-shot calls** — Each request triggers a chain of functions and returns a self-contained Div. Sections can run concurrently.
5. **Modularity (zero-trust)** — A file/section/service does not know and does not care how the next on-chain function handles information. Exception: shared `ServiceRequest` format.
6. **Functional first** — No OOP where it isn't needed. Plain function calls are fine. OOP is fine when the framework demands it (e.g., NautilusTrader `CoveredCallStrategy(Strategy)`).
7. **Readable** — Teammates must be able to understand the code and follow along. If things get too abstract, simplify.
8. **Docstrings** — On all relevant functions.
9. **Try-except** — On all service entry points and anywhere failure should be logged rather than crash.
10. **Minimal dependencies** — FastHTML, MonsterUI, FastAPI, NautilusTrader, Pydantic, LightGBM, scikit-learn, joblib. Do not add new deps without explicit approval.
11. **UI blocks during requests** — Temporary blocks on interactive elements until the previous response finishes rendering. If something fails, a generic fallback Div keeps the chain going.

---

## Strategy & Backtesting

The `CoveredCallStrategy` in `src/inference/strategy.py` is the **shared core** — it powers both:
- The **backtesting engine** (NautilusTrader `BacktestEngine`)
- The **backtesting dashboards** (UI visualization of strategy results)

### Design Principles

- **Modular**: The strategy is composable — individual components (model selection, moneyness, maturity, entry/exit rules) can be swapped independently.
- **UI-configurable**: Users will be able to modify strategy parameters from the trading screen (ambitious, but that's the target).
- **Plug-and-play models**: The inference service supports multiple models. Teammates (Swathi, Fatimat) are training new ones — integrating a better-performing model should require minimal code changes.
- **Easy to follow**: Anyone on the team should be able to read the strategy code and understand the decision logic without jumping through abstractions.

### Current Models

Located in `models/`. Best two will be wired into the inference service with UI toggle:

| Model | File | Macro F1 |
|-------|------|----------|
| LightGBM walk-forward | `lgbm_walkforward_daily.joblib` | 0.47 |
| LightGBM tuned | `lgbm_tuned_3class.joblib` | 0.35 |
| LSTM daily | `lstm_3class_daily.pt` | 0.41 |
| XGBoost tuned | `xgb_tuned_3class.joblib` | 0.34 |
| RF tuned | `rf_tuned_3class.joblib` | 0.34 |

---

## Ticker Universe

```python
UNIVERSE = ["AAPL", "AMZN", "AVGO", "GOOG", "GOOGL", "META", "MSFT", "NVDA", "TSLA", "WMT"]
```

---

## Data Paths

```
data/processed/
    daily_clean.parquet          — cleaned daily OHLCV (all tickers)
    options_clean.parquet        — cleaned options chains
    modeling_data.parquet        — daily features + labels (29K rows)
    monthly_labels.parquet       — monthly labels (1,391 rows)

models/
    lgbm_walkforward_daily.joblib
    lgbm_tuned_3class.joblib
    improved_model_metadata.json

reports/figures/                 — all PNGs for /docs screen
```

---

## Python Conventions

- **Type hints** on function signatures (no strict mypy enforcement).
- **Docstrings** on all public functions.
- **Try-except with logging** on all service entry points.
- **Ruff** for linting (`ruff check`).
- **No testing framework** — happy-path diagnostic scripts only. If it breaks on the happy path, log it and fix it.
- **Informal git commits** — no conventional commit enforcement.

---

## Security

- Never hardcode API keys, tokens, or secrets in source files.
- Never commit `.env`, credentials, or key files.
- S3 bucket URL for public read-only data is fine in code.
- Alpha Vantage API key must stay in environment variables only.

---

## What's NOT in Scope

- CI/CD pipelines
- Test suites or TDD
- Database integration
- JavaScript of any kind
- Over-engineered abstractions or premature optimization
