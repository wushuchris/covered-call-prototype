# CLAUDE.md - Covered Call Prototype

## Project Overview

ML-Driven Covered Call Decision Support for AAI-590 Capstone (USD / Validex Growth Investors).
Decision-support diagnostic tool, not an automated trading signal. Predicts which moneyness bucket (ATM, OTM5, OTM10) yields the best covered call return each month for 10 large-cap U.S. stocks, then presents analysis alongside strategy scoring, market context, and model limitations.

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
├── data/                        - computed parquets (bucket_returns), capstone_insights.md
├── ui/
│   ├── app.py                   - FastHTML :8008 (routes, USD branding, local fonts)
│   ├── ui_components.py         - MonsterUI component tree (launcher, diagnostic, docs screens)
│   ├── ui_handler.py            - aiohttp bridge to inference service
│   ├── ui_utils.py              - pack_request, send_to_inference
│   └── static/                  - logo, fonts, capstone PDF (served locally)
└── inference/
    ├── app.py                   - FastAPI :8009 (all endpoints including /scoring, /context, /claude_analysis)
    ├── graph.py                 - Graph 1: LangGraph DAG for dual-model inference (LGBM + LSTM-CNN parallel)
    ├── scoring_graph.py         - Graph 2: dual-model strategy scoring (Baseline, Argmax, Risk-Adjusted, Conservative)
    ├── context_graph.py         - Graph 3: market context (price regime, features, model track record)
    ├── analysis_graph.py        - Graph 4: Claude AI analysis (Haiku API, OVERVIEW + RECOMMENDED ACTION)
    ├── claude_analysis.py       - legacy stub (superseded by analysis_graph.py)
    ├── model.py                 - LGBM loading, feature store, predict_bucket(), compute_model_metrics()
    ├── lstm_model.py            - LSTM-CNN loading, feature store, predict_bucket(), get_monthly_predictions()
    ├── live_data.py             - yfinance + Black-Scholes + FRED for live dates beyond feature store range
    ├── daily.py                 - single-day + batch inference orchestration
    ├── chart.py                 - OHLC candlestick data builder
    ├── strategy.py              - simulate_inference() wrapper
    ├── scoring.py               - composable scoring engine (confidence, TC, delta-hedge)
    ├── backtesting.py           - backtest loop + caching (dual-model, Conservative only)
    ├── mlflow_reader.py         - reads mlruns/ flat-file backend directly
    ├── inference_utils.py       - input validation
    ├── capstone_insights.md     - extracted report insights for Claude analysis context
    └── nautilus_reference.py    - commented-out NautilusTrader class (reference only)
```

### Data Flow (Four-Graph Pipeline)

```
User sidebar -> POST /inference_call -> ui_handler -> aiohttp POST :8009/inference
    -> Graph 1 (inference): validate -> [LGBM, LSTM-CNN] parallel -> aggregate
    -> JSON response -> inference_results_card() -> htmx swap (immediate)

Then automatically (hx-trigger="load" on render):
    -> GET /claude_analysis -> runs Graph 2 + Graph 3 + Graph 4 sequentially:
        Graph 2 (scoring): load_data -> [baseline, lgbm_strategies, lstm_strategies] parallel -> aggregate
        Graph 3 (context): [price_context, feature_context, track_record] parallel -> aggregate
        Graph 4 (Claude): build_prompt -> call_claude_haiku -> format_response
    -> JSON response -> claude_analysis_card() -> htmx swap into #claude-analysis
```

UI renders progressively: model predictions first, then scoring + context + Claude analysis.

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
10. **Minimal dependencies** - FastHTML, MonsterUI, FastAPI, Pydantic, LightGBM, scikit-learn, joblib, torch, langgraph, anthropic, python-dotenv, yfinance, scipy, fredapi. Do not add new deps without explicit approval.
11. **UI blocks during requests** - Temporary blocks on interactive elements until the previous response finishes rendering. If something fails, a generic fallback Div keeps the chain going.

---

## Strategy & Backtesting

The scoring engine in `src/inference/scoring.py` sits between model predictions and trading decisions. NautilusTrader was evaluated and removed - the backtest loop runs in plain Python.

### Scoring Engine

Three weighted score components per ticker per month:
1. **Model Confidence** - LGBM prediction probability
2. **Transaction Cost** - bid-ask spread + turnover penalty
3. **Delta-Hedged Return** - vol premium after removing directional exposure

### Strategy Preset

| Preset | Confidence | TC | Delta-Hedge | Positions | Sizing |
|--------|-----------|-----|-------------|-----------|--------|
| Conservative | 30% | 50% | 20% | 7 | Equal |

Balanced and Aggressive presets were removed. Conservative is the production preset.
All strategies run on both LGBM and LSTM-CNN predictions separately.

### Additional Strategies (no scoring)

- **Argmax** - model's top pick per ticker, all tickers, equal weight
- **Risk-Adjusted** - P(bucket) × E[return|bucket], expanding historical averages, all tickers, equal weight
- **Baseline** - OTM10 short-dated on all tickers, equal weight (no model)

### Production Model

| Model | File | Test Macro F1 | Status |
|-------|------|---------------|--------|
| LightGBM walk-forward | `lgbm_3class_moneyness.joblib` | 0.47 (walk-forward) / 0.59 (2025 test) | Production (secondary) |
| LSTM-CNN best | `lstm_cnn_best_model.pth` | 0.11 (7-class) | Production (primary) - same as Streamlit |

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
    feature_store.parquet        - LGBM precomputed predictions (1,391 rows)
    lstm_feature_store.parquet   - LSTM-CNN precomputed predictions (~37K rows)
    lstm_scaler.joblib           - StandardScaler for live LSTM inference

models/
    lgbm_3class_moneyness.joblib   - production model (34 features, walk-forward trained)
    improved_model_metadata.json   - stale (lists 27 features, model uses 34 via model.feature_name_)

saved_models/
    lstm_cnn_best_model.pth        - team's LSTM-CNN (7-class, deployed on Streamlit)

final_datasets/
    daily_stock_optimal_bucket_modeling_with_fred.parquet - LSTM source data (41K daily rows, 35 features + FRED)

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
- **Informal git commits** - no conventional commit enforcement. No Co-Authored-By tags.
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
