# Carlos Ortiz — Session Log & Living Document
**Project**: ML-Driven Covered Call Optimization (AAI-590 Capstone)
**Branch**: `carlos`
**Role**: Infrastructure & utilities — building reproducible pipelines, project scaffolding, and support tooling for the team.

---

## Current State (as of 2026-03-07)

### Timeline Position
- **Week 2** (March 1-7): Project Architecture & Pipeline Design
- Week 3 starts March 8: Data Cleaning & EDA

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

### What's Next (Carlos's Queue)
- [x] Submit Kosmos query for feature/label engineering exploration — RUNNING
- [ ] Review Kosmos results when complete (~12 hrs)
- [ ] Present Idea 1 (contract-level dataset) + Kosmos findings to team for decision
- [ ] Build feature engineering pipeline (from raw parquets to modeling dataset)
- [ ] Align quarterly fundamentals to monthly decision points (merge_asof)
- [ ] Label construction (pending team decision + Kosmos results)
- [ ] EDA notebook for data quality and feature distributions
- [ ] Baseline model pipeline (Random Forest, then MLP, then Transformer)
- [ ] Backtesting simulation framework

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

### Key File Paths
```
notebooks/01_data_pull.ipynb          — data ingestion (done)
data_scripts/data_dwnld.py           — S3 quick-download (done)
persistence_agent/feature_inventory.xlsx — feature definitions (reference)
environment.yaml                      — conda environment spec
requirements-dev.txt                  — pip requirements
```
