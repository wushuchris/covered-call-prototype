# Covered Call Prototype

**AAI-590 Capstone — Validex Growth Investors**

An AI/ML prototype for optimizing covered call strike and maturity selection using fundamentals, valuation metrics, and market regime features.

---

## Table of Contents

1. [Overview](#overview)
2. [Stock Universe](#stock-universe)
3. [Environment Setup](#environment-setup)
4. [API Keys](#api-keys)
5. [Data Download](#data-download)
6. [Repository Structure](#repository-structure)
7. [Pipeline — notebooks/](#pipeline--notebooks)
8. [Experimental Work — notebooks-2/](#experimental-work--notebooks-2)
9. [Feature Groups](#feature-groups)
10. [Target Variable](#target-variable)
11. [Models](#models)
12. [Model Comparison](#model-comparison)
13. [Data Splits](#data-splits)
14. [EDA Highlights](#eda-highlights)
15. [Dependencies](#dependencies)
16. [Project Context](#project-context)
17. [License](#license)

---

## Overview

A covered call strategy involves selling (shorting) call options on existing stock positions to generate premium income while capping upside potential. Selecting the right strike and expiration depends on whether the stock is in a growth regime or a distress regime:

| Regime | Signal | Recommended Strategy |
|--------|--------|----------------------|
| Growth / Momentum | Strong revenue growth, expanding margins | OTM 10% calls — keep upside |
| Neutral | Stable fundamentals, low volatility | OTM 5% calls — balance premium vs. upside |
| Distress | Negative growth, declining cash, high leverage | ATM calls — maximize premium collection |

This prototype trains classifiers to predict the **optimal bucket** (strike + days-to-expiration) for each stock on each trading day.

---

## Stock Universe

10 large-cap tech / retail stocks with data from **2000 – present**:

`AAPL` · `MSFT` · `NVDA` · `AMZN` · `GOOG` · `GOOGL` · `META` · `TSLA` · `WMT` · `AVGO`

---

## Environment Setup

### Prerequisites

- [Anaconda](https://www.anaconda.com/download) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed
- Git installed

### Step 1 — Clone the repository

```bash
git clone <repo-url>
cd covered-call-prototype
```

### Step 2 — Create the conda environment

The full environment spec is in `environment.yaml`. The environment name is `covered_call_ml` and uses **Python 3.10**.

```bash
conda env create -f environment.yaml
```

This installs all conda and pip dependencies in one step, including:
- Core data science: `numpy`, `pandas`, `scipy`
- Machine learning: `scikit-learn`, `xgboost`, `lightgbm`, `optuna`, `shap`
- Deep learning: `pytorch`, `torchvision`, `torchaudio`, `transformers`, `accelerate`
- Finance / time series: `ta-lib`, `yfinance`, `alpha_vantage`, `fredapi`, `statsmodels`
- Visualisation: `matplotlib`, `seaborn`, `plotly`
- Utilities: `pyarrow`, `python-dotenv`, `tqdm`, `joblib`, `category-encoders`

> **Note:** The first run may take 5–10 minutes depending on your internet connection.

### Step 3 — Activate the environment

```bash
conda activate covered_call_ml
```

You should see `(covered_call_ml)` in your terminal prompt.

### Step 4 — Launch JupyterLab

```bash
jupyter lab
```

Your browser will open at `http://localhost:8888`. Navigate to the `notebooks/` folder and run them in order (`01` → `07`).

### Updating the environment (if dependencies change)

```bash
conda env update -f environment.yaml --prune
```

### Removing the environment

```bash
conda deactivate
conda env remove -n covered_call_ml
```

---

## API Keys

Two external APIs are required. Create a `.env` file in the project root (this file is git-ignored and must never be committed):

```
ALPHAVANTAGE_API_KEY=your_key_here
FRED_API_KEY=your_key_here
```

**Where to get them:**

| API | Purpose | Free tier |
|-----|---------|-----------|
| Alpha Vantage | Stock prices, options, fundamentals | https://www.alphavantage.co/support/#api-key |
| FRED | Federal Reserve economic indicators | https://fred.stlouisfed.org/docs/api/api_key.html |

> **Tip:** The free Alpha Vantage tier has rate limits (25 requests/day). Use the S3 download below to skip this step entirely.

---

## Data Download

### Option A — Download pre-built datasets from S3 (recommended)

Skip the API pull step entirely by downloading pre-cleaned datasets from the Validex S3 bucket:

```bash
python data_scripts/data_dwnld.py
```

This downloads:
- `ALL_daily_adjusted.parquet` — OHLCV prices
- `ALL_income_statement.parquet` — Income statement fundamentals
- `ALL_balance_sheet.parquet` — Balance sheet fundamentals
- `ALL_cash_flow.parquet` — Cash flow statements
- `ALL_options.parquet` — Options chain data
- `ALL_overview.csv` — Company overview metadata

### Option B — Pull fresh data from APIs

Run `notebooks/01_data_pull.ipynb` with your `.env` keys configured. This pulls directly from Alpha Vantage and FRED for the full ticker universe.

---

## Repository Structure

```
covered-call-prototype/
├── notebooks/                               # Primary pipeline (run in order)
│   ├── 01_data_pull.ipynb                  # Pull data from Alpha Vantage & FRED APIs
│   ├── 02_data_cleaning.ipynb              # Null handling, standardisation, S3 upload
│   ├── 03_eda.ipynb                        # Exploratory data analysis (39 plots)
│   ├── 03b_bucket_boundary_analysis.ipynb  # Strike / maturity bucket definitions
│   ├── 04_feature_engineering.ipynb        # Ratio engineering & data merging
│   ├── 05_label_construction.ipynb         # Target variable construction
│   ├── 06_baseline_model.ipynb             # XGBoost gradient-boosting classifier (Optuna)
│   ├── 06b_improved_models.ipynb           # Enhanced model variants (RF, LightGBM)
│   ├── 06c_improved_models.ipynb           # Further model improvements & two-stage approach
│   ├── 07_lstm_model.ipynb                 # LSTM deep learning classifier
│   ├── 08_model_comparison.ipynb           # Model comparison (placeholder)
│   ├── 09_optimization_layer.ipynb         # Optimization layer (placeholder)
│   ├── 10_backtesting.ipynb                # Backtesting framework (placeholder)
│   └── read_s3_data.ipynb                  # S3 data download utility
│
├── notebooks-2/                             # Experimental pipeline — FRED macro integration
│   ├── 01_data_pull.ipynb                  # Data extraction (variant)
│   ├── 02_data_clean.ipynb                 # Data cleaning (variant)
│   ├── 03_extensive_eda.ipynb              # Extended EDA with macro indicators
│   ├── 04_files combine_feature_engg.ipynb # Feature engineering with FRED macro features
│   ├── 05_data_modelling.ipynb             # Data modelling with macro-enriched dataset
│   ├── 06_XGBoost_baseline.ipynb           # XGBoost baseline on macro-enriched features
│   ├── 07_PatchTSTfor_Classification.ipynb # PatchTST transformer (100-day sequences)
│   └── 08_pretrained_patchTST_transformer.ipynb # Pre-trained PatchTST fine-tuning experiment
│
├── src/
│   ├── models/
│   │   ├── xgboost_model.py
│   │   └── lstm_model.py
│   ├── features/
│   │   └── build_features.py
│   └── evaluation/
│       └── backtest.py
├── data/
│   ├── raw/                                 # Raw API downloads (options by ticker)
│   ├── clean/                               # Cleaned parquet files
│   │   ├── daily_adjusted/
│   │   ├── fundamentals/
│   │   ├── options/
│   │   └── fred/
│   └── final/                               # Final ML-ready parquet files
├── final_datasets/
│   ├── daily_stock_optimal_bucket_modeling.parquet
│   ├── daily_stock_optimal_bucket_modeling_with_fred.parquet
│   ├── daily_stock_optimal_bucket_two_stage_modeling.parquet
│   └── master_price_fundamentals_features.parquet
├── model_datasets/                          # Scaled train / val / test (single-stage)
│   ├── X_train_scaled.parquet
│   ├── X_val_scaled.parquet
│   ├── X_test_scaled.parquet
│   ├── y_train.parquet
│   ├── y_val.parquet
│   └── y_test.parquet
├── models/                                  # Serialised trained models
│   ├── xgboost.joblib
│   ├── xgb_tuned_6class.joblib
│   ├── random_forest.joblib
│   ├── rf_tuned_6class.joblib
│   ├── lgbm_tuned_6class.joblib
│   ├── lgbm_3class_moneyness.joblib
│   ├── twostage_moneyness.joblib
│   ├── twostage_maturity.joblib
│   ├── lstm_6class.pt
│   ├── lstm_moneyness.pt
│   ├── lstm_maturity.pt
│   ├── pretrain_patchtst_full_model.pt
│   ├── pretrained_patchtst_best_model_with_fred.pth
│   ├── scaler.joblib
│   └── improved_model_metadata.json
├── eda_plots/                               # 39 PNG visualisations from EDA
├── saved_artifacts/                         # Features, models, and parameters storage
├── config/
│   └── config.yaml
├── data_scripts/
│   └── data_dwnld.py                       # Download pre-built datasets from S3
├── reports/
│   └── figures/
├── environment.yaml                         # Conda environment spec (Python 3.10)
├── requirements.txt
├── .env                                    # API keys — NOT committed
└── LICENSE
```

---

## Pipeline — notebooks/

The primary pipeline runs end-to-end in the `notebooks/` folder. Run notebooks in order:

```
01_data_pull          Pull raw OHLCV, fundamentals, options, FRED data
       ↓
02_data_cleaning      Handle nulls, forward-fill, standardise formats
       ↓
03_eda                Visualise price, volume, fundamentals, valuation (39 plots)
       ↓
03b_bucket_analysis   Define and validate strike/maturity bucket labels
       ↓
04_feature_eng        Engineer ratios, merge all sources into one dataset
       ↓
05_label_construction Construct target variables, time-based train/val/test split
       ↓
06_baseline_model     XGBoost classifier (Optuna tuning, SHAP importance)
       ↓
06b / 06c             Random Forest, LightGBM, two-stage model variants
       ↓
07_lstm_model         LSTM sequence classifier on rolling windows
```

### Notebook summaries

| Notebook | Key Actions | Output |
|----------|------------|--------|
| `01_data_pull` | Alpha Vantage API calls, FRED pull | `data/raw/` |
| `02_data_cleaning` | Null audit, ffill/bfill, split/dividend adjustments | `data/clean/` |
| `03_eda` | 39 visualisations across price, fundamentals, valuation | `eda_plots/` |
| `03b_bucket_boundary_analysis` | Bucket label definitions and boundary validation | Label schema |
| `04_feature_engineering` | Ratio engineering, quarterly → daily merge, FRED join | `master_price_fundamentals_features.parquet` |
| `05_label_construction` | Target variable creation, time-based split, StandardScaler | `model_datasets/` |
| `06_baseline_model` | Optuna-tuned XGBoost, class balancing, SHAP importance | `xgb_tuned_6class.joblib` |
| `06b_improved_models` | Random Forest and LightGBM variants | `rf_tuned_6class.joblib`, `lgbm_tuned_6class.joblib` |
| `06c_improved_models` | Two-stage moneyness + maturity models | `twostage_moneyness.joblib`, `twostage_maturity.joblib` |
| `07_lstm_model` | Rolling-window LSTM (6-class, moneyness, maturity) | `lstm_6class.pt`, `lstm_moneyness.pt`, `lstm_maturity.pt` |

---

## Experimental Work — notebooks-2/

The `notebooks-2/` folder contains a parallel experimental pipeline that explored whether **FRED macroeconomic indicators** (Fed funds rate, unemployment, VIX, yield curve) could improve model performance beyond stock-level fundamentals alone.

### What was tried

| Notebook | Description |
|----------|-------------|
| `01_data_pull` | Data extraction (mirrors primary pipeline) |
| `02_data_clean` | Data cleaning variant |
| `03_extensive_eda` | Extended EDA incorporating macro indicator analysis |
| `04_files combine_feature_engg` | Feature engineering with FRED macro features merged in |
| `05_data_modelling` | Modelling on macro-enriched dataset |
| `06_XGBoost_baseline` | XGBoost trained on features including FRED indicators |
| `07_PatchTSTfor_Classification` | PatchTST transformer on 100-day sequences with macro features |
| `08_pretrained_patchTST_transformer` | Pre-trained PatchTST fine-tuned on the covered-call task |

### Key finding

> **Adding FRED macroeconomic indicators did not meaningfully improve classification performance.** Despite enriching the feature set with Fed funds rate, unemployment, yield curve spreads, and VIX-related signals, model accuracy and macro-F1 scores remained similar to the stock-fundamentals-only baseline. This suggests that the covered-call bucket decision is driven primarily by company-specific regime signals rather than broad macroeconomic context — at least at the daily classification granularity used here.

The trained artefacts from this experiment are saved at:
- `models/pretrain_patchtst_full_model.pt` — PatchTST trained from scratch
- `models/pretrained_patchtst_best_model_with_fred.pth` — PatchTST fine-tuned with FRED features
- `final_datasets/daily_stock_optimal_bucket_modeling_with_fred.parquet` — FRED-enriched dataset

---

## Feature Groups

| Category | Examples |
|----------|---------|
| Price & Market | Daily returns, rolling volatility, volume trends |
| Profitability | Gross / operating / net margin, EBITDA |
| Growth | Revenue growth YoY, net income trend |
| Liquidity | Current ratio, cash runway, free cash flow |
| Leverage | Debt-to-equity, long-term debt / EBITDA |
| Valuation | P/S, EV/EBITDA, P/E, market cap |
| Economic Regime (FRED) | Fed funds rate, unemployment, VIX, yield curve spreads |

The PatchTST model uses the **top 50 features** ranked by XGBoost feature importance.

---

## Target Variable

**Single-stage:** predict a joint `Strike_DTE` bucket.

| Bucket | Meaning |
|--------|---------|
| `ATM_30` | At-the-money call, 30 days to expiration |
| `OTM5_60` | 5% out-of-the-money call, 60 DTE |
| `OTM10_60` | 10% out-of-the-money call, 60 DTE |
| `OTM10_90` | 10% out-of-the-money call, 90 DTE |

**Two-stage (experimental, `notebooks/06c_improved_models.ipynb`):**

1. **Stage 1 — Moneyness:** ATM · OTM5 · OTM10
2. **Stage 2 — Duration:** 30 DTE · 60 DTE · 90+ DTE

---

## Models

### Tree-based Models

| Model | File | Description |
|-------|------|-------------|
| XGBoost (baseline) | `xgboost.joblib` | Base gradient-boosted classifier |
| XGBoost (tuned) | `xgb_tuned_6class.joblib` | Optuna-tuned, class-balanced, 6-class |
| Random Forest (baseline) | `random_forest.joblib` | Ensemble baseline |
| Random Forest (tuned) | `rf_tuned_6class.joblib` | Tuned RF for 6-class prediction |
| LightGBM (tuned) | `lgbm_tuned_6class.joblib` | LightGBM 6-class classifier |
| LightGBM (moneyness) | `lgbm_3class_moneyness.joblib` | 3-class moneyness stage |

### Two-Stage Models

| Model | File | Description |
|-------|------|-------------|
| Stage 1 — Moneyness | `twostage_moneyness.joblib` | Classifies ATM / OTM5 / OTM10 |
| Stage 2 — Maturity | `twostage_maturity.joblib` | Classifies 30 / 60 / 90+ DTE |

### Deep Learning Models

| Model | File | Description |
|-------|------|-------------|
| LSTM (6-class) | `lstm_6class.pt` | Sequence classifier for joint Strike_DTE bucket |
| LSTM (moneyness) | `lstm_moneyness.pt` | LSTM for moneyness stage |
| LSTM (maturity) | `lstm_maturity.pt` | LSTM for maturity stage |
| PatchTST | `pretrain_patchtst_full_model.pt` | Transformer on 100-day sequences |
| PatchTST + FRED | `pretrained_patchtst_best_model_with_fred.pth` | PatchTST fine-tuned with macro features |

### PatchTST Architecture

- **Architecture:** PatchTST — divides time series into fixed-length patches and applies self-attention
- **Input window:** 100 trading days per sample
- **Sequences:** built per ticker with no look-ahead bias
- **Training:** PyTorch (GPU recommended)
- **Benefit over LSTM:** captures long-range dependencies, parallelisable training

---

## Model Comparison

All metrics are reported on the **held-out test set** (2024-01-01 onwards) unless noted. Models are grouped by experiment context since they were trained on slightly different task formulations (9-class, 6-class, 3-class moneyness, two-stage).

---

### Baseline Models — 9-Class (notebooks/06_baseline_model.ipynb)

Direct joint prediction of moneyness + maturity bucket (9 classes) using time-split cross-validation.

| Model | Test Accuracy | Macro F1 | Weighted F1 | CV Macro F1 |
|-------|:---:|:---:|:---:|:---:|
| Random Forest | 0.3728 | 0.3581 | 0.3626 | 0.3732 ± 0.014 |
| XGBoost | 0.3871 | 0.3556 | 0.3810 | 0.3695 ± 0.021 |

---

### Improved Models — 6-Class (notebooks/06b_improved_models.ipynb)

Reduced to 6 joint Strike_DTE buckets with Optuna hyperparameter tuning.

| Model | Test Accuracy | Macro F1 | Weighted F1 | CV Macro F1 |
|-------|:---:|:---:|:---:|:---:|
| Random Forest (baseline) | 0.2770 | 0.2244 | — | — |
| Random Forest (tuned) | 0.2850 | 0.2261 | 0.2848 | 0.2445 |
| XGBoost (tuned) | 0.2401 | 0.1928 | 0.2283 | 0.2519 |
| LightGBM (tuned) | 0.2164 | 0.1765 | 0.2169 | 0.2604 |
| Two-Stage (combined) | 0.2296 | 0.1667 | 0.2000 | — |

---

### Walk-Forward Validation — 3-Class Moneyness (notebooks/06c_improved_models.ipynb)

Decomposed the problem into moneyness-only (ATM / OTM5 / OTM10) using an expanding-window walk-forward evaluation. Significantly better generalisation than direct 6-class prediction.

| Model | Accuracy | Macro F1 | Weighted F1 | Top-2 Accuracy |
|-------|:---:|:---:|:---:|:---:|
| LightGBM | **0.5384** | **0.4736** | 0.5283 | **0.8097** |
| Random Forest | 0.5384 | 0.4733 | — | — |
| XGBoost | 0.5333 | 0.4656 | — | — |
| Maturity rule (IV rank) | 0.6190 | — | — | — |

---

### LSTM with Attention (notebooks/07_lstm_model.ipynb)

Sequence-based classifiers using rolling 3-day windows with attention.

| Model | Test Accuracy | Macro F1 | Weighted F1 |
|-------|:---:|:---:|:---:|
| LSTM — 6-class (direct) | 0.3846 | 0.3636 | 0.3899 |
| Two-Stage LSTM — combined | **0.4212** | **0.3839** | **0.4220** |
| └─ Stage 1: Moneyness (3-class) | 0.5900 | 0.5651 | — |
| └─ Stage 2: Maturity (2-class) | 0.6900 | 0.6488 | — |

---

### XGBoost with FRED Macro Features (notebooks-2/06_XGBoost_baseline.ipynb)

7-class classifier trained on stock fundamentals + FRED macroeconomic indicators.

| Split | Accuracy | Macro F1 | Balanced Accuracy |
|-------|:---:|:---:|:---:|
| Train | 0.8248 | 0.8298 | 0.8514 |
| Validation | 0.5391 | 0.2297 | 0.2423 |
| **Test** | **0.3331** | **0.1422** | **0.1766** |

> Large train–test gap indicates overfitting; FRED features did not improve generalisation.

---

### PatchTST Transformer (notebooks-2/07_PatchTSTfor_Classification.ipynb)

Transformer classifier on 100-day sequences — 7-class prediction with and without threshold tuning.

| Configuration | Val Accuracy | Val Macro F1 | Test Accuracy | Test Macro F1 |
|---------------|:---:|:---:|:---:|:---:|
| Baseline | 0.3695 | 0.1688 | 0.2498 | 0.1334 |
| Threshold-tuned | — | **0.1904** | 0.2689 | 0.1292 |

---

### Pre-trained PatchTST + FRED Fine-tuning (notebooks-2/08_pretrained_patchTST_transformer.ipynb)

Transfer learning experiment: pre-trained PatchTST backbone fine-tuned on the covered-call task with FRED-enriched features.

| Fine-tuning Stage | Val Accuracy | Val Macro F1 | Val Balanced Acc | Test Macro F1 |
|-------------------|:---:|:---:|:---:|:---:|
| Head-only tuning | — | 0.1332 | — | — |
| Full fine-tuning | 0.3259 | 0.1584 | 0.1991 | **0.1202** |

---

### Summary — Best Results per Model Family

| Model Family | Best Configuration | Test Accuracy | Test Macro F1 | Notes |
|---|---|:---:|:---:|---|
| Tree-based (baseline) | XGBoost 9-class | 0.387 | 0.356 | Best simple baseline |
| Tree-based (walk-forward) | LightGBM moneyness | **0.538** | **0.474** | Best overall macro F1 |
| Tree-based (FRED) | XGBoost 7-class | 0.333 | 0.142 | Overfit; FRED did not help |
| Two-stage tree | LightGBM moneyness + rule maturity | 0.538 | 0.474 | Decomposition helped |
| LSTM | Two-stage LSTM | 0.421 | 0.384 | Best deep-learning result |
| PatchTST (scratch) | Threshold-tuned | 0.269 | 0.133 | Limited by data size |
| PatchTST (fine-tuned) | Full fine-tune + FRED | 0.326 | 0.120 | No benefit from pre-training |

**Key takeaways:**
- Decomposing the problem into **moneyness + maturity stages** consistently outperformed direct joint bucket prediction
- **LightGBM with walk-forward validation** achieved the best generalisation (Macro F1 0.474, Top-2 Accuracy 0.81)
- **FRED macroeconomic features did not improve** any model — the task is driven by company-specific signals
- **Transformer models underperformed** relative to tree-based methods, likely due to limited training data for sequence modelling
- Class imbalance remains the primary challenge; macro F1 is the most meaningful metric for this task

---

## Data Splits

All splits are **time-based** to prevent look-ahead bias:

| Split | Date Range | Purpose |
|-------|-----------|---------|
| Train | Before 2022-01-01 | Model fitting |
| Validation | 2022-01-01 – 2023-12-31 | Hyperparameter tuning |
| Test | 2024-01-01 onwards | Final evaluation |

`StandardScaler` is **fit on the training set only** and applied to validation and test sets.

---

## EDA Highlights

39 visualisations are saved in `eda_plots/`:

| Series | Content |
|--------|---------|
| D-series | Price history, return distributions, rolling volatility, volume |
| F-series | Revenue growth, profitability margins, R&D spending, EBITDA trends |
| B-series | Assets vs. liabilities, D/E ratio, current ratio, cash position |
| C-series | Operating / investing / financing cash flows, FCF margin |
| O-series | Valuation multiples (P/S, EV/EBITDA, P/E), ROA / ROE |

**Key findings:**

- Revenue growth rate is the strongest single signal for regime classification
- Profitability transitions (loss-making → profitable) represent major strategy shifts
- Cash runway relative to burn rate is an early distress indicator
- Leverage context (D/E + cash flow coverage) matters more than leverage level alone
- AAPL/MSFT/GOOGL show profitability consistency that warrants OTM10 strategies; high-growth or loss-making names warrant ATM premium collection

---

## Dependencies

All packages are specified in `environment.yaml` (conda-forge + pip).

| Category | Packages |
|----------|---------|
| Data | `pandas`, `numpy`, `scipy`, `pyarrow` |
| ML | `scikit-learn`, `xgboost`, `lightgbm`, `optuna`, `shap`, `category-encoders` |
| Deep Learning | `torch`, `torchvision`, `torchaudio`, `transformers`, `accelerate` |
| Finance | `ta-lib`, `yfinance`, `alpha_vantage`, `fredapi`, `statsmodels`, `pandas-datareader` |
| Visualisation | `matplotlib`, `seaborn`, `plotly` |
| Utilities | `python-dotenv`, `requests`, `tqdm`, `joblib`, `pyyaml` |
| Notebook | `jupyter`, `jupyterlab`, `ipykernel` |

---

## Project Context

This prototype is developed as part of the **AAI-590 Capstone** course at the University of San Diego in collaboration with **Validex Growth Investors**. The goal is to automate covered call selection decisions that traditionally rely on analyst discretion, using a combination of structured financial data and sequence-aware deep learning.

---

## License

See [LICENSE](LICENSE).
