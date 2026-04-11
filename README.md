# Covered Call Strategy Classifier

**AAI-590 Capstone — Validex Growth Investors**

An AI/ML system that classifies the optimal covered call strike and maturity bucket for a given stock on a given day, using price history, financial fundamentals, valuation metrics, and market regime features. The best model is deployed as a FastAPI inference endpoint on AWS EC2, with a Streamlit UI and MLflow model registry.

**Live deployment (EC2):**
- Streamlit UI: http://98.93.2.225:8501
- FastAPI docs: http://98.93.2.225:8000/docs
- API health: http://98.93.2.225:8000/health
- MLflow registry: http://98.93.2.225:5000

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Stock Universe & Data](#2-stock-universe--data)
3. [Repository Structure](#3-repository-structure)
4. [ML Pipeline — Final Notebooks](#4-ml-pipeline--final-notebooks)
5. [Experimental Work — experiment_notebooks](#5-experimental-work--experiment_notebooks)
6. [Models & Results](#6-models--results)
7. [Backtesting Results](#7-backtesting-results)
8. [Deployment Architecture](#8-deployment-architecture)
9. [Quick Start — Run Locally](#9-quick-start--run-locally)
10. [Deploy to AWS EC2](#10-deploy-to-aws-ec2)
11. [Register Model in MLflow](#11-register-model-in-mlflow)
12. [API Reference](#12-api-reference)
13. [Environment Setup — Training](#13-environment-setup--training)
14. [Feature Groups](#14-feature-groups)
15. [Target Variable](#15-target-variable)
16. [Data Splits](#16-data-splits)

---

## 1. Project Overview

A covered call is an options strategy where you hold a stock and sell a call option against it to generate income. The key decisions are:

- **Moneyness**: Sell At-The-Money (ATM), 5% Out-of-The-Money (OTM5), or 10% Out-of-The-Money (OTM10)
- **Expiry**: 30-day, 60-day, or 90-day expiration

This project trains classifiers to predict the **optimal bucket** — the combination of strike and maturity that historically maximised risk-adjusted return — given today's market features.

**7 target classes:**

| Class | Strike | Expiry | Strategy |
|-------|--------|--------|----------|
| `ATM_30` | At-the-money | 30 days | Max premium, high assignment risk |
| `ATM_60` | At-the-money | 60 days | Balanced premium + time |
| `ATM_90` | At-the-money | 90 days | Lower theta, more time value |
| `OTM5_30` | 5% OTM | 30 days | Keep some upside, 30-day cycle |
| `OTM5_60_90` | 5% OTM | 60-90 days | Balanced upside + income |
| `OTM10_30` | 10% OTM | 30 days | Aggressive upside retention |
| `OTM10_60_90` | 10% OTM | 60-90 days | Max upside participation |

---

## 2. Stock Universe & Data

**10 Large-Cap US Stocks:** AAPL, MSFT, NVDA, AMZN, GOOG, GOOGL, META, TSLA, WMT, AVGO

**Data Sources:**
- **Alpha Vantage API** — Daily OHLCV prices, quarterly income statements, balance sheets, cash flows, options chains
- **FRED API** — Fed funds rate, unemployment, yield curve, VIX (macro features — tested but did not improve model performance)

**Date Range:** 2000-2025 | Training: pre-2022 | Validation: 2022-2023 | Test: 2024+

---

## 3. Repository Structure

```
covered-call-prototype/
├── final_notebooks/          # Primary ML pipeline — run in order
│   ├── 01_data_pull.ipynb
│   ├── 02_data_clean.ipynb
│   ├── 03_extensive_eda.ipynb
│   ├── 03b_bucket_boundary_analysis.ipynb
│   ├── 04_files combine_feature_engg.ipynb
│   ├── 05_data_modelling.ipynb
│   ├── 06_XGBoost_baseline.ipynb
│   ├── 07_PatchTSTfor_Classification.ipynb
│   ├── 07b_patchtst_walkforward.ipynb
│   ├── 07c_pretrained_patchTST_transformer.ipynb
│   ├── 08_LSTM_CNN_Classification.ipynb
│   ├── 08b_LSTM_CNN_Classification_regularized.ipynb
│   ├── 08c_LSTM_CNN_Classification 3.ipynb
│   └── 10_mlflow_tracking.ipynb
│
├── experiment_notebooks/     # Exploratory development notebooks
│   ├── 02_data_cleaning.ipynb
│   ├── 03_eda.ipynb
│   ├── 03b_bucket_boundary_analysis.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_label_construction.ipynb
│   ├── 06_baseline_model.ipynb
│   ├── 06b_improved_models.ipynb
│   ├── 06c_improved_models.ipynb
│   ├── 07_lstm_model.ipynb
│   ├── 08_model_comparison.ipynb
│   ├── 09_optimization_layer.ipynb
│   └── 10_backtesting.ipynb
│
├── app/                      # Deployable application
│   ├── docker-compose.yml    # Orchestrates all 3 services
│   ├── api/
│   │   ├── main.py           # FastAPI inference server
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   ├── frontend/
│   │   ├── app.py            # Streamlit UI
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   └── model/
│       └── lstm_cnn.py       # Shared model class (API + frontend)
│
├── saved_models/             # Production model checkpoints
│   ├── lstm_cnn_best_model.pth
│   └── log_to_mlflow.py      # Script to register model in MLflow
│
├── infra/
│   └── terraform-app/        # AWS EC2 deployment via Terraform
│       ├── main.tf
│       ├── variables.tf
│       ├── outputs.tf
│       ├── app_bootstrap.sh
│       └── terraform.tfvars.example
│
├── final_datasets/           # ML-ready parquet files
├── models/                   # All trained model checkpoints (30+)
├── model_datasets/           # Scaled train/val/test splits
├── environment.yaml          # Conda environment spec
└── data_scripts/
    └── data_dwnld.py         # Download raw data from S3
```

---

## 4. ML Pipeline — Final Notebooks

Run notebooks in numerical order. Each reads from the previous step's output.

### 01 — Data Acquisition (`01_data_pull.ipynb`)
Pulls raw data from Alpha Vantage and FRED APIs for all 10 tickers (2000-2025).
- **Requires:** `ALPHA_VANTAGE_API_KEY` and `FRED_API_KEY` in `.env`
- **Output:** `data/raw/` — daily OHLCV, quarterly fundamentals, options chains, FRED series

### 02 — Data Cleaning (`02_data_clean.ipynb`)
Null audit, forward-fill, split/dividend adjustment, standardisation, parquet conversion.
- **Output:** `data/clean/*.parquet`

### 03 — EDA & Bucket Analysis (`03_extensive_eda.ipynb` + `03b`)
39+ visualisations across price history, returns, margins, cash flows, volatility regimes.

Defines delta-based covered call bucket boundaries:
- ATM: delta 0.45-0.60 (mid 0.52)
- OTM5: delta 0.30-0.45 (mid 0.37)
- OTM10: delta 0.15-0.30 (mid 0.22)

### 04 — Feature Engineering (`04_files combine_feature_engg.ipynb`)
Merges daily prices with quarterly fundamentals (forward-filled to daily). Engineers 53 features across price, technical, momentum, valuation, profitability, leverage, growth, and macro categories.
- **Output:** `final_datasets/master_price_fundamentals_features.parquet`

### 05 — Target Construction & Splits (`05_data_modelling.ipynb`)
Creates `optimal_bucket` labels across 7 classes. Applies time-based train/val/test split. Fits StandardScaler on training data only.
- **Output:** `model_datasets/X_train/val/test_scaled.parquet`, `y_*.parquet`
- **Output:** `final_datasets/daily_stock_optimal_bucket_modeling.parquet`

### 06 — XGBoost Baseline (`06_XGBoost_baseline.ipynb`)
XGBoost 7-class classifier with Optuna hyperparameter tuning and partial class resampling.

**Results:** Test Accuracy 0.333 | Macro F1 0.142 | Balanced Accuracy 0.177

**Conclusion:** Direct 7-class joint prediction suffers from severe class imbalance; model predicts heavily toward dominant classes.

### 07 — PatchTST Transformer (`07_PatchTSTfor_Classification.ipynb`, `07b`, `07c`)
Patch-based time series transformer on 100-day sliding window sequences.
- `07b` — Walk-forward annual retraining, RF-selected top-35 features
- `07c` — PatchTST fine-tuned with FRED macro features (no improvement)
- **Result:** Macro F1 0.133 — transformer underperformed; insufficient data for attention to generalise

### 08 — LSTM-CNN Classifier (`08_LSTM_CNN_Classification.ipynb`, `08b`, `08c`)
Hybrid deep learning model combining a 2-layer CNN branch for local temporal pattern extraction with a bidirectional LSTM branch capturing long-range dependencies, fused via Bahdanau attention. Uses 50-day rolling window sequences and top-35 features.

- `08b` — L2 regularisation and dropout tuning
- `08c` — Optuna hyperparameter refinement, final best checkpoint

Best checkpoint params: `cnn_out_channels=128, kernel_size=7, lstm_hidden=128, lstm_layers=2, attn_dim=128, dropout=0.155`

**Output:** `saved_models/lstm_cnn_best_model.pth` — production deployment model

### 10 — MLflow Tracking (`10_mlflow_tracking.ipynb`)
Logs all trained models to MLflow, registers the best LSTM-CNN as `CoveredCallLSTMCNN` with the `Champion` alias.
- **Requires:** MLflow server running at http://localhost:5000
- **Output:** `mlruns/` tracking data, model registry

---

## 5. Experimental Work — experiment_notebooks

These notebooks document the iterative development that shaped the final pipeline. They use a simpler **3-class moneyness formulation** (ATM / OTM5 / OTM10) without maturity decomposition.

### `02-04` — Data & Feature Prototypes
Early data cleaning, EDA, and feature engineering prototypes. Established the delta-based bucket boundaries and feature set adopted in the final pipeline.

### `05_label_construction.ipynb`
3-class moneyness label construction — identifies the optimal strike bucket per (symbol, month) based on risk-adjusted covered call return from actual options data.

### `06_baseline_model.ipynb` + `06b` + `06c`
Progressive model development:
- `06` — Random Forest and XGBoost baselines with time-split CV
- `06b` — Optuna tuning, added LightGBM, class weights
- `06c` — Walk-forward annual retraining, implied volatility features from options data

**Key finding:** LightGBM with walk-forward validation achieved **Macro F1 0.474** on 3-class moneyness — the best generalisation across all models.

### `07_lstm_model.ipynb`
LSTM with 60-day lookback on 3-class problem. Walk-forward validation (2014-2025). **Result:** F1 0.384.

### `08_model_comparison.ipynb`
Side-by-side comparison of RF, XGBoost, LightGBM, and LSTM. LightGBM walk-forward was the clear winner.

### `09_optimization_layer.ipynb`
Probability-based strike selection on top of LightGBM. Three selection modes:
- **Argmax** — pick highest probability class
- **Risk-adjusted** — weight by probability x expected return
- **Diversified** — blend across buckets proportional to probability

Produces monthly `strike_recommendations.csv` used in backtesting.

### `10_backtesting.ipynb`
Full covered call backtest simulation (2014-2025) using actual options data. Compares model-guided vs static benchmarks across 10 tickers, 128 months, 724 trades. See [Backtesting Results](#7-backtesting-results).

---

## 6. Models & Results

| Model | Task | Metric | Result |
|-------|------|--------|--------|
| XGBoost (tuned) | 7-class direct | Macro F1 | 0.142 |
| Random Forest (tuned) | 3-class moneyness | Balanced Accuracy | 0.37-0.54 |
| **LightGBM (walk-forward)** | **3-class moneyness** | **Macro F1** | **0.474** |
| Two-Stage LightGBM | moneyness + maturity | Combined F1 | 0.474 |
| LSTM (walk-forward) | 3-class | F1 | 0.384 |
| LSTM-CNN + Attention | 7-class direct | — | Deployed model |
| PatchTST | 7-class direct | Macro F1 | 0.133 |
| PatchTST + FRED | 7-class direct | Macro F1 | 0.120 |

**Key findings:**
- LightGBM walk-forward on 3-class moneyness = best generalisation (Macro F1 0.474)
- FRED macro features did not improve any model
- Transformer models underperformed tree-based methods — insufficient data volume
- Direct 7-class joint prediction is harder than decomposed 3-class due to class imbalance
- LSTM-CNN is the deployed production model — end-to-end deep learning inference

### LSTM-CNN Architecture (Deployed Model)

```
Input: (batch, seq_len=50, n_features=35)
         |
         +---> CNN Branch
         |     Conv1d(35 -> 128, kernel=7) -> BatchNorm -> ReLU -> Dropout
         |     Conv1d(128 -> 128, kernel=7) -> BatchNorm -> ReLU -> Dropout
         |     AdaptiveAvgPool1d(1)  ->  (batch, 128)
         |
         +---> BiLSTM Branch (2 layers, hidden=128)
         |     Bidirectional LSTM  ->  (batch, seq_len, 256)
         |     Bahdanau Attention  ->  (batch, 256)
         |
         +---> Fusion
               Concat  ->  (batch, 384)
               LayerNorm -> Dropout
               FC(384 -> 192) -> ReLU -> Dropout
               FC(192 -> 7)
```

---

## 7. Backtesting Results

**Period:** Feb 2014 - Dec 2025 | **128 months** | **10 tickers** | **724 trades**

Simulation uses actual options data (bid/ask/mark prices). Equal-weight portfolio, monthly rebalancing.

### Risk Metrics (sorted by Sharpe)

| Strategy | Ann. Return | Sharpe | Max Drawdown | Win Rate |
|----------|------------|--------|-------------|---------|
| Oracle (theoretical best) | 338.2% | 5.82 | -8.4% | 97.7% |
| Always-ATM | 313.9% | 5.36 | -9.2% | 97.7% |
| Model-Risk-Adjusted | 284.2% | 5.07 | -11.3% | 96.9% |
| Model-Argmax | 282.4% | 5.05 | -12.7% | 96.9% |
| Model-Diversified | 249.7% | 4.90 | -13.8% | 96.1% |
| Always-OTM5 | 155.5% | 4.46 | -12.2% | 92.2% |
| Random | 162.3% | 4.27 | -11.3% | 92.2% |
| Always-OTM10 | 73.1% | 2.90 | -17.0% | 81.3% |

### Statistical Significance (paired t-test vs Always-ATM)

| Model | Mean Monthly Alpha | p-value | Significant? |
|-------|-------------------|---------|-------------|
| Model-Argmax | -0.0074 | 0.0014 | Yes |
| Model-Risk-Adj | -0.0070 | 0.0023 | Yes |

### Interpretation
- Model strategies achieve Sharpe ~5.05-5.07 vs Always-ATM at 5.36
- The model underperforms Always-ATM by ~0.7% per month (statistically significant but small)
- Always-ATM is a very strong benchmark in sustained bull markets
- Oracle Sharpe of 5.82 confirms headroom exists for a stronger classifier to beat Always-ATM
- Model adds value during regime transitions; richer features or larger dataset may close the gap

**Limitations:** No transaction costs or slippage modelled; equal-weight allocation; monthly rebalancing.

---

## 8. Deployment Architecture

```
                    +--------------------------------------+
                    |         AWS EC2 t3.medium            |
                    |                                      |
  Browser --------->  Streamlit  :8501                    |
                    |      |                               |
                    |      v                               |
                    |  FastAPI   :8000  <-- /predict       |
                    |      |                               |
                    |      v                               |
                    |  MLflow    :5000  (model registry)   |
                    +--------------------------------------+
```

| Service | Image | Port | Purpose |
|---------|-------|------|---------|
| `mlflow` | `ghcr.io/mlflow/mlflow:v2.13.0` | 5000 | Model registry + experiment tracking |
| `api` | Built from `app/api/Dockerfile` | 8000 | FastAPI inference server |
| `frontend` | Built from `app/frontend/Dockerfile` | 8501 | Streamlit user interface |

**Model load priority:**
1. MLflow Model Registry (`models:/CoveredCallLSTMCNN/Champion`) — if registered
2. Local file fallback (`/app/saved_models/lstm_cnn_best_model.pth`) — always works

---

## 9. Quick Start — Run Locally

### Prerequisites
- Docker Desktop installed and running

### Steps

```bash
git clone https://github.com/wushuchris/covered-call-prototype.git
cd covered-call-prototype/app
docker-compose up --build
```

Wait 3-5 minutes. When healthy:

| Service | URL |
|---------|-----|
| Streamlit UI | http://localhost:8501 |
| FastAPI docs | http://localhost:8000/docs |
| MLflow UI | http://localhost:5000 |

```bash
# Verify
curl http://localhost:8000/health

# Stop
docker-compose down
```

---

## 10. Deploy to AWS EC2

### Prerequisites

```bash
brew install awscli hashicorp/tap/terraform
aws configure   # Access Key ID, Secret Key, us-east-1, json
```

### Step 1 — Create EC2 Key Pair
1. AWS Console -> EC2 -> Key Pairs -> Create key pair
2. Name it (e.g. `covered-call-key`), format `.pem`
3. `mv ~/Downloads/covered-call-key.pem ~/.ssh/ && chmod 400 ~/.ssh/covered-call-key.pem`

### Step 2 — Find Your IP
```bash
curl https://checkip.amazonaws.com
# e.g. 130.45.13.198  ->  use as  130.45.13.198/32
```

### Step 3 — Configure Terraform
```bash
cd infra/terraform-app
cp terraform.tfvars.example terraform.tfvars
nano terraform.tfvars
```

```hcl
aws_region    = "us-east-1"
instance_type = "t3.medium"
key_pair_name = "covered-call-key"       # key pair name (no .pem)
your_ip_cidr  = "130.45.13.198/32"       # your IP from Step 2
repo_url      = "https://github.com/wushuchris/covered-call-prototype"
```

### Step 4 — Provision
```bash
terraform init
terraform apply -auto-approve
```

Example output:
```
api_docs_url       = "http://98.93.2.225:8000/docs"
api_url            = "http://98.93.2.225:8000"
instance_public_ip = "98.93.2.225"
mlflow_url         = "http://98.93.2.225:5050"
ssh_command        = "ssh -i ~/.ssh/covered-call-gpu-key.pem ec2-user@98.93.2.225"
streamlit_url      = "http://98.93.2.225:8501"
```

### Step 5 — Wait for Bootstrap (4-6 minutes)
```bash
ssh -i ~/.ssh/covered-call-key.pem ec2-user@98.93.2.225
tail -f /var/log/app_bootstrap.log
# Wait for: === Bootstrap complete ===
```

### Step 6 — Verify
```bash
cd /home/ec2-user/covered-call-prototype/app
docker-compose ps        # all 3 should show: healthy
curl http://localhost:8000/health
```

### Step 7 — Access
| Service | URL |
|---------|-----|
| Streamlit UI | http://98.93.2.225:8501 |
| FastAPI docs | http://98.93.2.225:8000/docs |
| API health | http://98.93.2.225:8000/health |
| MLflow UI | http://98.93.2.225:5000 |
| SSH | `ssh -i ~/.ssh/covered-call-gpu-key.pem ec2-user@98.93.2.225` |

### Update after code changes
```bash
ssh -i ~/.ssh/covered-call-gpu-key.pem ec2-user@98.93.2.225
cd /home/ec2-user/covered-call-prototype
git pull origin main
cd app
docker-compose build --no-cache api
docker-compose up -d
```

### Tear down
```bash
cd infra/terraform-app
terraform destroy -auto-approve
```

---

## 11. Register Model in MLflow

Once containers are running, register the model so the API loads from the registry.

### Step 1 — Log and register
```bash
conda activate covered_call_ml

# Register in EC2 MLflow
python saved_models/log_to_mlflow.py \
    --model_path saved_models/lstm_cnn_best_model.pth \
    --tracking_uri http://98.93.2.225:5000

# OR register in local Docker MLflow
python saved_models/log_to_mlflow.py \
    --model_path saved_models/lstm_cnn_best_model.pth \
    --tracking_uri http://localhost:5000
```

### Step 2 — Set Champion alias
1. Open MLflow UI: http://98.93.2.225:5000 (or http://localhost:5000)
2. Models -> CoveredCallLSTMCNN -> click version -> Add/Edit Aliases -> type `Champion` -> Save

### Step 3 — Restart API
```bash
# On EC2
cd /home/ec2-user/covered-call-prototype/app
docker-compose restart api
docker-compose logs api --tail=20
# Should show: Model loaded from MLflow: models:/CoveredCallLSTMCNN/Champion
```

---

## 12. API Reference

Base URL: `http://98.93.2.225:8000` (EC2) or `http://localhost:8000` (local)

### `GET /health`
```bash
curl http://98.93.2.225:8000/health
```
```json
{
  "status": "ok",
  "device": "cpu",
  "model_source": "file",
  "seq_len": 50,
  "n_features": 35,
  "n_classes": 7
}
```

### `GET /model/info`
Returns feature names, class labels, best params, per-class thresholds.

### `POST /predict`
Input shape: `[N, seq_len=50, n_features=35]`

```bash
curl -X POST http://98.93.2.225:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sequences": [[[0.1, 0.2, ...], ...]]}'
```

```json
{
  "model_source": "file",
  "n_predictions": 1,
  "results": [{
    "predicted_class": "OTM5_60_90",
    "probabilities": {"ATM_30": 0.05, "OTM5_60_90": 0.42, "...": "..."},
    "confidence": 0.42
  }]
}
```

### `POST /predict/csv`
Upload CSV with `symbol`, `date`, and feature columns. Auto-builds 50-day sliding windows per symbol.

```bash
curl -X POST http://98.93.2.225:8000/predict/csv \
  -F "file=@your_stock_data.csv"
```

**Interactive docs:** http://98.93.2.225:8000/docs

---

## 13. Environment Setup — Training

```bash
conda env create -f environment.yaml
conda activate covered_call_ml

cp .env.example .env
# Edit .env: ALPHA_VANTAGE_API_KEY and FRED_API_KEY

jupyter lab
```

### Download datasets from S3
```bash
python data_scripts/data_dwnld.py
```

---

## 14. Feature Groups

| Group | Features |
|-------|----------|
| Price & Returns | Open, High, Low, Close, Volume, daily/weekly/monthly returns |
| Technical | Rolling volatility (5/20/60d), momentum, RSI, MACD, Bollinger Bands |
| Valuation | P/E, P/S, EV/EBITDA, Price/Book |
| Profitability | Gross/Operating/Net margin, ROA, ROE |
| Leverage | Debt/Equity, Interest coverage, Current ratio |
| Growth | YoY revenue growth, EPS growth, FCF growth |
| Macro (experimental) | Fed funds rate, unemployment, yield curve, VIX |

Top 35 features selected by Random Forest importance — used by LSTM-CNN and PatchTST.

---

## 15. Target Variable

`optimal_bucket` = covered call strategy that maximised risk-adjusted return on a given day:

```
optimal_bucket = argmax over (moneyness x maturity) of risk_adjusted_return(t, t+expiry)
```

| Moneyness | Delta range | Mid delta |
|-----------|------------|-----------|
| ATM | 0.45-0.60 | 0.52 |
| OTM5 | 0.30-0.45 | 0.37 |
| OTM10 | 0.15-0.30 | 0.22 |

OTM5_60 and OTM5_90 merged to `OTM5_60_90`; OTM10_60 and OTM10_90 merged to `OTM10_60_90` due to low frequency. Class distribution is imbalanced — handled via class weights and focal loss in LSTM-CNN.

---

## 16. Data Splits

| Split | Date Range | Purpose |
|-------|-----------|---------|
| Train | 2005-2021 | Model fitting |
| Validation | 2022-2023 | Hyperparameter tuning |
| Test | 2024-2025 | Final held-out evaluation |

StandardScaler fitted on training data only, applied to all splits.

---

## License

See [LICENSE](LICENSE).
