# Carlos Ortiz — Session Log & Living Document
**Project**: ML-Driven Covered Call Optimization (AAI-590 Capstone)
**Branch**: `carlos`
**Role**: Infrastructure & utilities — building reproducible pipelines, project scaffolding, and support tooling for the team.

---

## Current State (as of 2026-03-15)

### Timeline Position
- **Week 4** (March 15-21): Feature Engineering & Label Construction
- Carlos & Fatimat are primary contributors this week

### What Exists
| Component | Status | Owner | Notes |
|---|---|---|---|
| GitHub repo + branches | Done | Christopher | main, carlos, swathi branches active |
| Alpha Vantage data pull | Done | Christopher/Swathi | `notebooks/01_data_pull.ipynb` — pulls daily, fundamentals, options |
| S3 mirror of processed data | Done | Christopher | `validex-ml-data.s3.us-east-1.amazonaws.com` |
| Quick download script | Done | — | `data_scripts/data_dwnld.py` — reads parquets from S3 |
| Feature inventory | Done | Christopher | `persistence_agent/feature_inventory.xlsx` — 48 engineered features defined |
| Environment spec | Done | — | `environment.yaml` (conda) + `requirements-dev.txt` (pip) |
| Project management plan | Done | Team | `persistence_agent/AAI-590-G6 Project Management Plan.docx` |
| Project-wide utils | Done | Carlos | `src/utils.py` — logger, log_call decorator, shared ServiceRequest model |
| UI service (FastHTML) | Done | Carlos | `src/ui/` — launcher, trading screen, inference sidebar, backtesting dashboard, docs |
| Inference service (FastAPI) | Done | Carlos | `src/inference/` — daily inference, backtesting, strategy scaffold |
| Main launcher | Done | Carlos | `main.py` — spawns UI (8008) + inference (8009) as Popen processes |
| Python package structure | Done | Carlos | `__init__.py` files + `pyproject.toml` — `pip install -e .`, no sys.path hacks |
| USD branding | Done | Carlos | Local fonts (Sofia Sans EC, Spectral), Founders/Immaculata/Torero Blue, logo PNG |
| Systemd deployment | Done | Carlos | `deploy/setup_services.sh` — gunicorn + uvicorn workers, auto-start on boot |

### What's Next (Carlos's Queue)
- [x] Submit Kosmos query for feature/label engineering exploration — RUNNING
- [x] Build prototype microservices (UI + inference) — pseudocode → working code
- [x] Convert to proper Python packages (`__init__.py` + `pyproject.toml`)
- [x] USD branding (local fonts, brand colors, logo)
- [x] Systemd deployment script (gunicorn + uvicorn workers)
- [x] Add USD logo to `src/ui/static/`
- [ ] Review Kosmos results when complete (~12 hrs)
- [ ] Present Idea 1 (contract-level dataset) + Kosmos findings to team for decision
- [ ] Build feature engineering pipeline (from raw parquets to modeling dataset)
- [ ] Align quarterly fundamentals to monthly decision points (merge_asof)
- [ ] Label construction (pending team decision + Kosmos results)
- [ ] EDA notebook for data quality and feature distributions
- [ ] Baseline model pipeline (Random Forest, then MLP, then Transformer)
- [ ] Wire NautilusTrader engine into backtesting.py (venue, instruments, data loading)
- [ ] Plotly/ApexChart integration for candlestick chart in inference results panel

---

## Constraints & Decisions

### Hard Constraints
- **7-week timeline** — must ship by April 4, 2026
- **15 hrs/week per member** — scope accordingly
- **Alpha Vantage API** — rate-limited; data already cached in S3
- **10-ticker universe**: ADMA, NTRA, AXON, SHAK, AAPL, MSFT, NVDA, AMZN, GOOG, META
- **Monthly decision frequency** — one decision per ticker per month
- **9-class action space**: ATM/OTM5/OTM10 x 30/60/90 DTE

### Design Decisions
- Conventional ML libraries only (scikit-learn, XGBoost, LightGBM, PyTorch for MLP)
- Notebook-first development — every pipeline step must be reproducible in .ipynb
- Explainability required — SHAP, feature importances, ablation analysis
- Baseline comparison: always sell 30 DTE, 10% OTM calls

---

## Teammate Activity Log

### Christopher Mendoza
- Set up repo structure, drove initial data pull
- Created feature inventory spreadsheet
- Primary on Weeks 1-2 architecture

### Swathi Pabbathi
- First PR merged (branch: swathi)
- Data pull contributions, merged into main

### Fatimat Atanda
- Support role in early weeks, primary on Week 4 feature engineering

---

## Session Notes

### 2026-03-07 — Session 1: Project Bootstrap & Data Understanding
- Read entire codebase and all team documents
- Set up Claude Code auto-approval settings
- Created living document (this file) and project whitepaper (`WHITEPAPER.md`)
- Downloaded all datasets from S3 to `data/processed/`
- Deep research on Edison Scientific Kosmos (autonomous AI scientist) — architecture is transferable but the tool itself is science-domain-specific. Open-source reimplementation at github.com/jimmc414/Kosmos could be adapted. Academic free tier available via .edu email.
- TA-Lib feature research: identified 20 Tier-1 features (ADX, BBANDS, NATR, RSI, KAMA, HT_TRENDMODE) beyond the existing 48 in the feature inventory

**Key understandings established:**
- Options data = monthly SNAPSHOTS of the full options board (all listed contracts), NOT executed trades. ~900 calls per AAPL snapshot, only ~48 for ADMA.
- Final modeling dataset: ~1,320 rows (132 months × 10 tickers). Features computed on full daily history, then sampled at monthly decision dates. Temporal info is encoded into features (rolling windows), not preserved as sequences.
- Fundamentals join via merge_asof (most recent quarter ≤ decision date). Stay constant across 2-3 monthly rows until next quarter.
- Label construction is the hardest part: pick 9 representative contracts (one per bucket), simulate covered call payoff looking forward to expiration, best bucket = label.
- Panel data structure: 10 cross-sectional obs per date, ~132 independent time points. Train/test split by DATE cutoff, shuffle within train.
- 60/90 DTE labels have temporal overlap with subsequent decision dates — subtle leakage concern.
- Three model architectures: Random Forest (baseline), MLP (feedforward neural net), Transformer (tabular, e.g., FT-Transformer)

**Pain points identified for Kosmos query:**
- Feature engineering: which features actually matter? Risk of overfitting with ~90 features on ~1,320 rows
- Label engineering: how to define "best" bucket (raw return vs Sharpe vs downside-adjusted), handling missing bucket coverage for small-cap tickers (ADMA, NTRA), overlapping label windows for 60/90 DTE

- Kosmos query submitted to Edison Scientific (full run, 200 credits, .edu academic tier)
  - Uploaded: ALL_options.parquet, ALL_daily_adjusted.parquet, ALL_income_statement.parquet, ALL_balance_sheet.parquet, ALL_cash_flow.parquet
  - Query asks for: literature review on ML for covered calls/options, label transformation exploration, data quality filters, feature importance, regime stability
  - Query text saved at `persistence_agent/kosmos_query.md`
- **Next session priority**: Review Kosmos results, present findings + Idea 1 to team, begin feature engineering pipeline

### 2026-03-15 — Session 2: Pseudocode → Working Microservices

Turned all pseudocode specs under `src/` into working implementations. Original spec comments preserved above each function for team accountability.

**Built:**
- `src/utils.py` — `create_logger()` (console=ERROR, file=INFO, auto-cleanup >1 day), `log_call()` decorator (async-aware, logs name/file/duration), `ServiceRequest` Pydantic BaseModel (shared inter-service data structure)
- `src/ui/ui_utils.py` — `pack_request()`, `send_to_inference()` (async aiohttp POST to port 8009)
- `src/ui/ui_components.py` — launcher screen, trading screen (NavBar + daily inference + backtesting dashboards + docs footer), docs screen (Nymo-inspired sidebar + content layout). Uses proper MonsterUI API: `NavBar`, `Grid`, `Card(header=...)`, `Loading`, `Toast`, `TableFromDicts`, `Container`, `TextPresets`, `CheckboxX`, `NavContainer`
- `src/ui/ui_handler.py` — `handle_inference_call()` / `handle_backtest_call()` (pack → aiohttp POST → return)
- `src/ui/app.py` — FastHTML on port 8008. Routes: `/` (launcher), `/trading`, `/clear`, `/inference_call` (POST), `/docs`
- `src/inference/app.py` — FastAPI on port 8009. Routes: `/inference` (POST), `/backtest` (POST). Request body uses `ServiceRequest` Pydantic model (not raw dict)
- `src/inference/inference_utils.py` — `unpack_request()`, `validate_ticker()`, `validate_date()`
- `src/inference/strategy.py` — Placeholder functions (`random_bucket_strategy`, `baseline_strategy`, `simulate_inference`) + NautilusTrader `CoveredCallStrategy(Strategy)` scaffold with `on_start`, `on_bar`, `on_stop`
- `src/inference/backtesting.py` — Cache-first pattern (load/save JSON), placeholder report, `_build_engine()` NautilusTrader scaffold with production TODOs
- `src/inference/daily.py` — `run_daily_inference()` (validate → simulate)
- `main.py` — Popen launcher for both services, graceful Ctrl+C shutdown

**Sanity checks & fixes:**
- Read MonsterUI docs (llms.txt, llms-ctx.txt, dashboard source, API ref) — fixed `NavBarContainer` (doesn't exist) → `NavBar`, raw `uk-grid` → `Grid`, raw spinner → `Loading`, manual table → `TableFromDicts`, raw classes → `TextPresets`/`TextT`
- Read FastHTML docs (llms-ctx.txt, handler ref) — confirmed route param mapping, `serve()` usage
- Read FastAPI docs — fixed `body: dict` (won't parse as JSON body) → `body: ServiceRequest` (Pydantic BaseModel)
- Read NautilusTrader docs (getting started, strategy concepts, backtest low/high level) — scaffolded `CoveredCallStrategy(Strategy)` with proper lifecycle hooks, `BacktestEngine` setup in backtesting.py
- Eliminated cross-service import (inference was importing from ui) — moved `ServiceRequest` to shared `utils.py`
- Fixed relative cache path in backtesting.py → absolute via `Path(__file__).resolve()`

**Data flow:**
```
User sidebar → @rt POST → ui_handler.pack_request() → aiohttp POST (model_dump())
    → FastAPI parses ServiceRequest body → inference_utils.unpack_request()
    → daily.run_daily_inference() / backtesting.run_backtest()
    → JSON response → ui_handler → inference_results_card() → hx-swap into DOM
```

**Session 2 continued — packaging, branding, deployment:**

- Converted flat `src/` to proper Python packages: added `__init__.py` to `src/`, `src/ui/`, `src/inference/`; created `pyproject.toml` with runtime deps; all imports now `from src.xxx`; removed all `sys.path.insert` hacks; `pip install -e .` makes everything resolve
- Requirements coexistence: `pyproject.toml` (package + runtime deps), `carlos-reqs.txt` (Pi-light, includes `-e .`), `requirements-dev.txt` (team EDA/notebooks)
- Downloaded USD brand fonts locally (Sofia Sans Extra Condensed 600, Spectral 400/600) to `src/ui/static/fonts/` — no Google Fonts API calls, works offline
- Wired USD brand colors: Founders Blue `#003b70` (headings, navbar), Immaculata Blue `#0074c8` (buttons, links, accents), Torero Blue `#75bee9` (chart borders, subtle gradients)
- Downloaded USD logo PNG to `src/ui/static/usd_logo.png`
- Replaced JS checkbox with pure htmx (`hx_get="/today_date"` → server returns pre-filled date input)
- Added Validex Growth Investors branding to launcher and navbar
- Added `UkIcon("brain-circuit")` to navbar brand
- Docs screen: added "Back to Trader" link with `UkIcon("arrow-left")`
- Fixed `static_path` — FastHTML serves from `{static_path}/{url_path}`, so pointed to `src/ui/` dir (not `src/ui/static/`)
- Fixed full-width layout — removed `Container(ContainerT.xl)` cap, added CSS override for `.uk-container`/`.uk-section` max-width
- Results panel now always shows table structure with blank `—` values on load; Compute Inference swaps in real data
- Table and chart placeholder render side by side (flex 2:3 ratio)
- Created `deploy/setup_services.sh` — systemd units for gunicorn + uvicorn workers (2 workers each, `Type=simple`, bound to `0.0.0.0:8008`/`127.0.0.1:8009`, `PYTHONPATH` set, auto-restart on failure)

**Next session priority**: Review Kosmos results, wire real data into NautilusTrader engine, add candlestick chart (ApexChart/Plotly) to inference panel

### 2026-03-24/25 — Session 3: Model Integration, Candlestick Charts, Documentation

Picked up from a crashed session (computer turned off). Restored services, installed missing deps, wired the real LGBM model into the inference pipeline, added candlestick charts, built the full documentation site.

**Infrastructure fixes:**
- Installed `joblib`, `lightgbm`, `scikit-learn` (minimal ML deps — avoided the full `requirements-dev.txt` which crashed the Pi)
- Ran `data_scripts/build_processed.py` to generate `data/processed/` from S3 (1,391 rows, 9 classes, 10 tickers)
- Fixed inference service crash-loop caused by missing `joblib` import

**Model integration (new files):**
- `src/inference/model.py` — loads `lgbm_3class_moneyness.joblib`, builds a precomputed feature store from `modeling_data.parquet` + `options_clean.parquet` (IV features), runs predictions on all rows at startup. `predict_bucket(ticker, date)` is an instant row lookup.
- `src/inference/chart.py` — loads `daily_clean.parquet`, returns OHLC data (trailing month, ~23 candles) as list of dicts for ApexCharts
- `src/inference/strategy.py` — rewritten to delegate to `model.predict_bucket()` as single source of truth for both UI and NautilusTrader
- `src/inference/daily.py` — wired to call `simulate_inference()` + `build_candlestick_data()`

**Model behavior:**
- Exact month match first, then snaps to nearest available month if data is sparse (with `snapped: true` flag)
- UI shows a toast warning when snapping occurs
- Removed ground truth fields (best_bucket, best_return) from daily inference display — those belong in backtesting dashboards
- Added "LGBM 3-Class Moneyness" model name to results table

**Candlestick chart (ApexCharts, no JS):**
- Initially tried Plotly `to_html()` + `NotStr()` — htmx wouldn't execute the `<script>` tags
- Researched alternatives: `fh-plotly` pattern, kaleido SVG (segfaults on aarch64), matplotlib
- Final solution: MonsterUI's built-in `ApexChart` component with `apex_charts=True` in theme headers. `<uk-chart>` custom element self-initializes on htmx swap — zero JS authoring
- OHLC data returned as JSON from inference service, rendered as candlestick by UI component
- Limited to trailing month (~23 trading days) to keep browser responsive

**UI changes:**
- Renamed "Daily Inference" → "Historical Inference" (section heading + navbar)
- Updated ticker list to match actual universe (AAPL, AMZN, AVGO, GOOG, GOOGL, META, MSFT, NVDA, TSLA, WMT)
- Results table: Ticker, Date, Month, Model, Prediction, Confidence, Baseline, Sample
- Snap warning toast with `z-index:9999` to render above other components

**Documentation site (`/docs`):**
- Built full documentation from notebook markdown cells (01-07) and `reports/figures/` (21 PNGs)
- 6 sections: Overview, Data Pipeline, Exploratory Analysis, Feature Engineering, Models, Results
- Overview + Data Pipeline: full-width text + card layout
- Other sections: two-column layout — text left, modal trigger right (Lucide `search` icon opens full-size image in `Modal`)
- Horizontal divider between each row for visual clarity
- Scrollspy navbar with section anchors + "Trader" link back to trading screen
- Figures served via `Starlette Mount` at `/figures/` (route inserted at position 0 in `app.routes`) — no symlinks, portable across clones

**Logger fix:**
- Applied `@log_call(logger)` decorator to all UI routes — logger was empty because no routes were decorated
- Now logs timestamp, route, file, and execution time for every request

**Next session priority**: Backtesting dashboards (ground truth metrics belong there), NautilusTrader engine wiring, Kosmos results review

### 2026-04-03/04 — Session 4: Team Merge, Batch Inference, Dashboards, Docs Overhaul

Merged main branch (team's work) into carlos. Resolved conflicts in .gitignore and README. Read and analyzed the team's entire codebase: `app/` (Streamlit + FastAPI + Docker), `saved_models/` (LSTM-CNN checkpoints), `saved_artifacts/` (PatchTST, XGBoost), `mlruns/` (20 experiment runs), `final_notebooks/` (7-class pipeline).

**Key findings from team analysis:**
- Team built a separate Streamlit UI + Docker deployment for LSTM-CNN 7-class model
- Best LSTM-CNN: 0.110 macro F1 / 38.1% accuracy (regularised variant) — barely above random on 7-class
- Our LGBM 3-class: 0.59 F1 / 63% accuracy on test — significantly better
- The deployed Streamlit app serves the *worst* LSTM-CNN variant (0.091 F1, "Best" checkpoint)
- Two completely different pipelines: experiment notebooks (3-class, our work) vs final notebooks (7-class, team's work)

**NautilusTrader removal:**
- Confirmed NautilusTrader was pure dead weight — imported but never instantiated, no venues/instruments/orders
- Stripped from strategy.py, backtesting.py, pyproject.toml, carlos-reqs.txt
- Preserved as `src/inference/nautilus_reference.py` (commented out reference)

**Batch inference:**
- Added "All Stocks" checkbox to inference sidebar with htmx dropdown toggle (grays out ticker selection)
- Same button/route (`/inference_call`), branches internally on `batch` param
- New `/inference_batch` endpoint loops all 10 tickers, returns per-ticker results + summary stats
- Results panel shows summary card with expand icon → modal with all ticker rows
- Each row has search icon → nested modal with lazy-loaded candlestick chart
- Charts lazy-loaded via `hx-trigger="intersect once"` — solves ApexCharts hidden-container rendering issue

**UI fixes:**
- Replaced MonsterUI Toast with static `uk-alert` Div — Toast needs JS init that doesn't fire on htmx swap
- Snap warnings now display correctly for both single and batch inference
- Added tooltips (`uk-tooltip`) across all UI sections — plain English explanations on hover for:
  - Inference: Prediction, Confidence, Baseline, Sample type
  - Strategy: Annualized Return, Sharpe, Max Drawdown, Hit Rate, Avg P/L, Conservative/Balanced/Aggressive presets
  - Model Performance: Accuracy, Macro F1, Top-2, Precision, Recall, F1, Support, confidence analysis

**Dashboards — tabbed backtesting section:**
- Restructured backtesting section with UIKit `uk-tab` + `uk-switcher` (zero JS)
- Tab 1 — Strategy: existing backtest comparison, now with 6 columns (Baseline, Argmax, Risk-Adjusted, Conservative, Balanced, Aggressive)
- Tab 2 — Model Performance: summary metrics, per-class breakdown, confidence analysis, LGBM vs LSTM-CNN comparison, per-year table. Lazy-loaded. Sidebar filters (sample type, year) with mutual grayout.
- Tab 3 — MLflow: placeholder for experiment tracking

**Model Performance tab:**
- New `/model_metrics` endpoint computes live from feature store (not hardcoded)
- Renamed in-sample/out-of-sample → Train Dataset/Test Dataset throughout
- Default view shows Test Dataset (63% acc, 0.59 F1) not inflated All Data (97%)
- Filter toggles: Train/Test grays out year; specific year grays out sample type
- Train/test split confirmed: train = all data < 2025, test = 2025 (100 samples)

**Strategy backtest — Argmax and Risk-Adjusted columns:**
- Added `_compute_argmax_return()` — model's top pick per ticker, all tickers, equal weight
- Added `_compute_risk_adjusted_return()` — P(bucket) × E[return|bucket] with expanding historical averages (no lookahead)
- Stored full probability distribution (prob_ATM/OTM5/OTM10) in feature store for risk-adjusted scoring
- Risk-Adjusted achieved best Sharpe (2.69) and best max drawdown (-19.6%) across all strategies
- Budget input commented out — percentage metrics invariant to budget scale

**Feature store bug fix:**
- Discovered metadata listed 27 features but model trained on 34 (27 base - 5 dropped + 9 IV + 3 leaky removed)
- Old feature store masked the bug (loaded from cached .parquet)
- Fixed: now uses `model.feature_name_` directly instead of stale metadata

**Documentation overhaul:**
- Added sticky left sidebar with scrollspy navigation (Nymo whitepaper style)
- Restructured from 6 to 8 sections: Overview, Data Pipeline, EDA, Features, Tree-Based Pipeline, Deep Learning Pipeline, Results, Strategy & Post-Inference
- Features section: two-column layout (LGBM features left | DL features right)
- Results section: two-column layout (3-class metrics left | 7-class metrics right)
- Tree-Based Pipeline: full progression (RF → XGB → LGBM → walk-forward + LSTM 3-class comparison)
- Deep Learning Pipeline: XGBoost 7-class baseline → LSTM-CNN architecture (3 variants, 20 MLflow runs) → PatchTST → Docker deployment
- Strategy & Post-Inference: scoring engine documentation (3 components, 3 presets), backtesting results
- Audited all figures — removed misleading ones (broken Oracle cumulative returns, wrong feature importance placements, stale 9-class/6-class figures)

**Stub cleanup:**
- Removed empty stub directories from main merge: `src/models/`, `src/features/`, `src/evaluation/`

**Next session priority**: MLflow tab (serve experiment data on Pi), budget scoring redesign (make budget meaningful), notebook backtest strategy integration

---

## Proposed Dataset Redesign (TEAM DECISION REQUIRED)

### Idea 1: Contract-Level Dataset (Recommended)
**Status**: Proposal — needs team buy-in before implementation.

Instead of 1 row per (ticker, date) with a 9-class label, make 1 row per (ticker, date, contract):

| Aspect | Original Plan | Idea 1 |
|---|---|---|
| Grain | (ticker, month) | (ticker, month, contract) |
| Rows | ~1,026 | ~900K (calls only) |
| Label | 9-class: which bucket is best | Regression: realized covered call return for this contract |
| Features | ~90 stock-level features | ~90 stock features + ~10 contract features (moneyness, DTE, IV, delta, spread, OI) |
| Post-inference | Model picks bucket directly | Score every available contract, rank by predicted return, pick best |
| Bucket decision | Baked into label | Post-inference (flexible, can change buckets without retraining) |

**Why this is better:**
- 900K rows vs 1,026 — fundamentally different ML regime
- No arbitrary bucketing — model learns continuous relationship between contract characteristics and outcomes
- Contract features (IV, delta, spread) become inputs instead of being discarded
- Naturally handles uneven options coverage across tickers (ADMA 48 calls vs AAPL 900)
- Bucket strategy becomes a post-inference ranking decision, decoupled from the model

**Train/Val/Test split (with buffer, now affordable):**
```
Train:   2015 – Dec 2021     (~525K rows)
Buffer:  Jan–Mar 2022        (discard, prevents label→feature leakage at boundary)
Val:     Apr 2022 – Dec 2023 (~175K rows)
Buffer:  Jan–Mar 2024        (discard)
Test:    Apr 2024 – 2025     (~155K rows)
```
Still temporal cutoff + shuffle within each split.

**Open question — submitted to Kosmos:** What should the regression target be? No prescribed candidates — Kosmos explores freely.

### Idea 2: IV Surface Features (Complementary to either approach)
Extract features from the full options book per (ticker, date):

```
iv_atm_30d         — ATM implied vol at 30 DTE
iv_atm_90d         — ATM implied vol at 90 DTE
iv_term_slope      — 90d IV - 30d IV (term structure)
iv_skew_30d        — OTM put IV - ATM IV (skew)
iv_rank_90d        — current IV vs 90-day percentile
put_call_oi_ratio  — total put OI / total call OI
avg_call_spread    — mean bid-ask spread (liquidity proxy)
```

These features use the FULL options board (~900 contracts) to characterize market expectations, rather than discarding 99% of the data. Works with either the original 1,026-row dataset or the contract-level 900K-row dataset.

---

## Architecture Notes

### Data Flow
```
Alpha Vantage API → raw JSON (cached)
    ↓
Parsing + cleaning → processed parquets (per ticker + combined ALL_*)
    ↓
Feature engineering → modeling dataset (fundamentals + price + options features)
    ↓
Label construction → 9-class target (optimal covered call bucket)
    ↓
Train/val/test split (temporal) → model training
    ↓
Backtest simulation → performance vs baseline
```

### Service Architecture
```
Production (systemd):
    covered-call-ui.service        — gunicorn + uvicorn workers → src.ui.app:app :8008
    covered-call-inference.service — gunicorn + uvicorn workers → src.inference.app:app :8009
    nginx reverse proxy            — :80/:443 → :8008

Development:
    python main.py                 — Popen launcher for both services

src/ (Python package, pip install -e .)
    ├── __init__.py
    ├── utils.py               — logger, log_call decorator, ServiceRequest model
    ├── ui/
    │   ├── __init__.py
    │   ├── app.py             — FastHTML :8008 (routes, USD branding, local fonts)
    │   ├── ui_components.py   — MonsterUI component tree (Validex branding)
    │   ├── ui_handler.py      — aiohttp bridge to inference service
    │   ├── ui_utils.py        — pack_request, send_to_inference
    │   └── static/
    │       ├── usd_logo.png
    │       └── fonts/         — Sofia Sans EC 600, Spectral 400/600
    └── inference/
        ├── __init__.py
        ├── app.py             — FastAPI :8009 (inference, backtesting)
        ├── daily.py           — single-day paper-trade inference
        ├── backtesting.py     — NautilusTrader backtest engine (scaffold)
        ├── strategy.py        — CoveredCallStrategy + placeholder functions
        └── inference_utils.py — validation, data integrity
```

### Key File Paths
```
pyproject.toml                        — package definition + runtime deps
carlos-reqs.txt                       — Pi-light requirements (includes -e .)
main.py                               — dev launcher (Popen, both services)
deploy/setup_services.sh              — systemd unit setup script (gunicorn)
src/utils.py                          — project-wide utilities + shared ServiceRequest
src/ui/app.py                         — FastHTML server (port 8008)
src/ui/ui_components.py               — all UI components (MonsterUI + USD branding)
src/ui/ui_handler.py                  — inter-service communication handler
src/ui/ui_utils.py                    — request packing, async transport
src/ui/static/                        — logo, fonts (served locally, no API calls)
src/inference/app.py                  — FastAPI server (port 8009)
src/inference/strategy.py             — NautilusTrader strategy + placeholders
src/inference/backtesting.py          — backtest engine scaffold + caching
src/inference/daily.py                — daily inference pipeline
src/inference/inference_utils.py      — input validation
persistence_agent/                    — team docs, feature inventory, project plans, grounding
notebooks/01_data_pull.ipynb          — data ingestion (done)
data_scripts/data_dwnld.py           — S3 quick-download (done)
```
