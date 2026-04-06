# CLAUDE.md - Covered Call Prototype

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
main.py                          - dev launcher (Popen, both services)
deploy/setup_services.sh         - systemd (gunicorn + uvicorn), auto-start

src/
├── utils.py                     - logger, log_call decorator, ServiceRequest (shared Pydantic model)
├── data/                        - computed parquets (bucket_returns)
├── ui/
│   ├── app.py                   - FastHTML :8008 (routes, USD branding, local fonts)
│   ├── ui_components.py         - MonsterUI component tree (launcher, trading, docs screens)
│   ├── ui_handler.py            - aiohttp bridge to inference service
│   ├── ui_utils.py              - pack_request, send_to_inference
│   └── static/                  - logo, fonts (served locally, no external API calls)
└── inference/
    ├── app.py                   - FastAPI :8009 (/inference, /inference_batch, /backtest, /model_metrics, /chart_data)
    ├── model.py                 - model loading, feature store, predict_bucket(), compute_model_metrics()
    ├── daily.py                 - single-day + batch inference orchestration
    ├── chart.py                 - OHLC candlestick data builder
    ├── strategy.py              - simulate_inference() wrapper (NautilusTrader removed)
    ├── scoring.py               - composable scoring engine (confidence, TC, delta-hedge)
    ├── backtesting.py           - backtest loop + caching (Argmax, Risk-Adjusted, 3 presets)
    ├── inference_utils.py       - input validation
    └── nautilus_reference.py    - commented-out NautilusTrader class (reference only)
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

1. **Python only** - No JavaScript. FastHTML + MonsterUI handle all UI. ApexCharts via `<uk-chart>`.
2. **No databases** - Minimal side-effects/persistence. Parquet files and JSON caches only.
3. **Async first** - Don't block the event loop. Use `async def` for all service entry points and inter-service calls.
4. **One-shot calls** - Each request triggers a chain of functions and returns a self-contained Div. Sections can run concurrently.
5. **Modularity (zero-trust)** - A file/section/service does not know and does not care how the next on-chain function handles information. Exception: shared `ServiceRequest` format.
6. **Functional first** - No OOP where it isn't needed. Plain function calls are fine.
7. **Readable** - Teammates must be able to understand the code and follow along. If things get too abstract, simplify.
8. **Docstrings** - On all relevant functions.
9. **Try-except** - On all service entry points and anywhere failure should be logged rather than crash.
10. **Minimal dependencies** - FastHTML, MonsterUI, FastAPI, Pydantic, LightGBM, scikit-learn, joblib. Do not add new deps without explicit approval.
11. **UI blocks during requests** - Temporary blocks on interactive elements until the previous response finishes rendering. If something fails, a generic fallback Div keeps the chain going.

---

## Strategy & Backtesting

The scoring engine in `src/inference/scoring.py` sits between model predictions and trading decisions. NautilusTrader was evaluated and removed - the backtest loop runs in plain Python.

### Scoring Engine

Three weighted score components per ticker per month:
1. **Model Confidence** - LGBM prediction probability
2. **Transaction Cost** - bid-ask spread + turnover penalty
3. **Delta-Hedged Return** - vol premium after removing directional exposure

### Strategy Presets

| Preset | Confidence | TC | Delta-Hedge | Positions | Sizing |
|--------|-----------|-----|-------------|-----------|--------|
| Conservative | 30% | 50% | 20% | 7 | Equal |
| Balanced | 33% | 33% | 34% | 5 | Equal |
| Aggressive | 60% | 10% | 30% | 3 | Proportional |

### Additional Strategies (no scoring)

- **Argmax** - model's top pick per ticker, all tickers, equal weight
- **Risk-Adjusted** - P(bucket) × E[return|bucket], expanding historical averages, all tickers, equal weight
- **Baseline** - OTM10 short-dated on all tickers, equal weight (no model)

### Production Model

| Model | File | Test Macro F1 | Status |
|-------|------|---------------|--------|
| LightGBM walk-forward | `lgbm_3class_moneyness.joblib` | 0.47 (walk-forward) / 0.59 (2025 test) | Production |
| LSTM-CNN regularised | `lstm_cnn_best_model.pth` | 0.11 (7-class) | Dashboard only |

---

## Ticker Universe

```python
UNIVERSE = ["AAPL", "AMZN", "AVGO", "GOOG", "GOOGL", "META", "MSFT", "NVDA", "TSLA", "WMT"]
```

---

## Data Paths

```
data/processed/
    daily_clean.parquet          - cleaned daily OHLCV (all tickers)
    options_clean.parquet        - cleaned options chains
    modeling_data.parquet        - daily features + labels (29K rows)
    monthly_labels.parquet       - monthly labels (1,391 rows)

models/
    lgbm_3class_moneyness.joblib   - production model (34 features, walk-forward trained)
    improved_model_metadata.json   - stale (lists 27 features, model uses 34 via model.feature_name_)

saved_models/
    lstm_cnn_best_model.pth        - team's LSTM-CNN (7-class, deployed on Streamlit)

src/data/
    bucket_returns.parquet         - per-bucket realized returns for backtesting

reports/figures/                 - all PNGs for /docs screen
```

---

## Python Conventions

- **Type hints** on function signatures (no strict mypy enforcement).
- **Docstrings** on all public functions.
- **Try-except with logging** on all service entry points.
- **Ruff** for linting (`ruff check`).
- **No testing framework** - happy-path diagnostic scripts only. If it breaks on the happy path, log it and fix it.
- **Informal git commits** - no conventional commit enforcement.
- **No em dashes** - use hyphens (`-`) or rephrased sentences. Never use `—` in code, UI text, comments, or docs.

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
