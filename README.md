# Covered Call Strategy Classifier

**AAI-590 Capstone — Validex Growth Investors**

An AI/ML system that classifies the optimal covered call strike and maturity bucket for a given stock on a given day, using price history, financial fundamentals, valuation metrics, and market regime features.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Stock Universe & Data](#2-stock-universe--data)
3. [Repository Structure](#3-repository-structure)
4. [ML Pipeline — Final Notebooks](#4-ml-pipeline--final-notebooks)
5. [Models & Results](#5-models--results)
6. [Deployment Architecture](#6-deployment-architecture)
7. [Quick Start — Run Locally](#7-quick-start--run-locally)
8. [Deploy to AWS EC2](#8-deploy-to-aws-ec2)
9. [API Reference](#9-api-reference)
10. [MLflow Model Registry](#10-mlflow-model-registry)
11. [Environment Setup (Training)](#11-environment-setup-training)
12. [Feature Groups](#12-feature-groups)
13. [Target Variable](#13-target-variable)
14. [Data Splits](#14-data-splits)

---

## 1. Project Overview

A covered call is an options strategy where you hold stock and sell a call option against it to generate income. The key decisions are:

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
| `OTM5_60_90` | 5% OTM | 60–90 days | Balanced upside + income |
| `OTM10_30` | 10% OTM | 30 days | Aggressive upside retention |
| `OTM10_60_90` | 10% OTM | 60–90 days | Max upside participation |

---

## 2. Stock Universe & Data

**10 Large-Cap US Stocks:** AAPL, MSFT, NVDA, AMZN, GOOG, GOOGL, META, TSLA, WMT, AVGO

**Data Sources:**
- **Alpha Vantage API** — Daily OHLCV prices, quarterly income statements, balance sheets, cash flows, options chains
- **FRED API** — Fed funds rate, unemployment, yield curve, VIX (macro features — tested but did not improve performance)

**Date Range:** 2000–2025 | Training: pre-2022 | Validation: 2022–2023 | Test: 2024+

---

## 3. Repository Structure

```
covered-call-prototype/
├── final_notebooks/          # Full ML pipeline (run in order)
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
│   └── lstm_cnn_best_model.pth
│
├── infra/
│   └── terraform-app/        # AWS EC2 deployment (Terraform)
│       ├── main.tf
│       ├── variables.tf
│       ├── outputs.tf
│       ├── app_bootstrap.sh
│       └── terraform.tfvars.example
│
├── final_datasets/           # ML-ready parquet files
├── models/                   # All trained model checkpoints
├── model_datasets/           # Scaled train/val/test splits
├── environment.yaml          # Conda environment spec
└── data_scripts/
    └── data_dwnld.py         # Download raw data from S3
```

---

## 4. ML Pipeline — Final Notebooks

Run notebooks in numerical order. Each notebook reads from the previous step's output.

### Step 1 — Data Acquisition (`01_data_pull.ipynb`)
Pulls raw data from Alpha Vantage and FRED APIs for all 10 tickers.
- **Requires:** `ALPHA_VANTAGE_API_KEY` and `FRED_API_KEY` in `.env`
- **Output:** `data/raw/` — daily prices, quarterly fundamentals, options chains

### Step 2 — Data Cleaning (`02_data_clean.ipynb`)
Null audit, forward-fill, split/dividend adjustments, parquet conversion.
- **Output:** `data/clean/*.parquet`

### Step 3 — EDA (`03_extensive_eda.ipynb` + `03b_bucket_boundary_analysis.ipynb`)
39+ visualisations across price history, returns, margins, cash flows.
Defines delta-based bucket boundaries:
- ATM: delta 0.45–0.60
- OTM5: delta 0.30–0.45
- OTM10: delta 0.15–0.30

### Step 4 — Feature Engineering (`04_files combine_feature_engg.ipynb`)
Merges daily prices with quarterly fundamentals (forward-filled to daily), engineers financial ratios, and optionally joins FRED macro features.
- **Output:** `final_datasets/master_price_fundamentals_features.parquet`

### Step 5 — Target Construction & Splits (`05_data_modelling.ipynb`)
Creates `optimal_bucket` labels, applies time-based train/val/test split, fits StandardScaler on training data only.
- **Output:** `model_datasets/X_train/val/test_scaled.parquet`, `y_train/val/test.parquet`
- **Output:** `final_datasets/daily_stock_optimal_bucket_modeling.parquet`

### Step 6 — XGBoost Baseline (`06_XGBoost_baseline.ipynb`)
XGBoost 7-class classifier with Optuna hyperparameter tuning and class weighting.
- **Output:** `models/xgb_tuned_6class.joblib`, feature importance

### Step 7 — PatchTST Transformer (`07_PatchTSTfor_Classification.ipynb`, `07b`, `07c`)
Patch-based time series transformer with 100-day sliding window sequences.
- `07b` — Walk-forward annual retraining with RF-selected top-35 features
- `07c` — Fine-tuned PatchTST with FRED macro features (no improvement observed)
- **Output:** `models/patchtst_*.pt`, `models/pretrained_patchtst_*.pth`

### Step 8 — LSTM-CNN Classifier (`08_LSTM_CNN_Classification.ipynb`, `08b`, `08c`)
Hybrid CNN + BiLSTM + Bahdanau attention model with 50-day sequences.
- `08b` — L2 regularisation and dropout tuning
- `08c` — Further hyperparameter refinement
- **Output:** `models/lstm_*.pt`, `saved_models/lstm_cnn_best_model.pth`

### Step 9 — MLflow Tracking (`10_mlflow_tracking.ipynb`)
Logs all trained models (XGBoost, PatchTST, LSTM-CNN) to MLflow, registers the best LSTM-CNN as `CoveredCallLSTMCNN` with the `Champion` alias in the model registry.
- **Requires:** MLflow server running at `http://localhost:5000`
- **Output:** `mlruns/` tracking database, model registry

---

## 5. Models & Results

| Model | Task | Val Macro F1 | Notes |
|-------|------|-------------|-------|
| XGBoost (tuned) | 7-class | 0.14 | Baseline, class imbalance challenge |
| Random Forest | 6-class | 0.37–0.54 accuracy | Acceptable accuracy, poor minority recall |
| **LightGBM (walk-forward)** | **3-class moneyness** | **0.474** | **Best generalisation** |
| Two-Stage (LightGBM) | moneyness + maturity | 0.474 combined | Decomposed 6-class |
| LSTM | 6-class | 0.384 (two-stage) | RNN only, no CNN |
| LSTM-CNN + Attention | 7-class | — | Deployed model |
| PatchTST | 7-class | 0.133 | Transformer underperforms on small dataset |
| PatchTST + FRED | 7-class | 0.120 | Macro features did not help |

**Key findings:**
- LightGBM with walk-forward validation on 3-class moneyness achieved the best generalisation
- FRED macroeconomic features did **not** improve model performance
- Transformers underperformed tree-based methods — likely insufficient data volume
- The LSTM-CNN with attention is the deployed production model (end-to-end deep learning)

### LSTM-CNN Architecture

```
Input: (batch, seq_len=50, n_features)
         │
         ├──▶ CNN Branch
         │    Conv1d(kernel=7) → BatchNorm → ReLU → AdaptiveAvgPool
         │    → (batch, 128)
         │
         ├──▶ BiLSTM Branch (2 layers, hidden=128)
         │    Bidirectional LSTM → Bahdanau Attention
         │    → (batch, 256)
         │
         └──▶ Fusion
              Concat → LayerNorm → FC(384→192) → ReLU → FC(192→7)
```

---

## 6. Deployment Architecture

```
                    ┌──────────────────────────────────────┐
                    │            EC2 t3.medium              │
                    │                                      │
  Browser ─────────▶  Streamlit  :8501                    │
                    │      │                               │
                    │      ▼                               │
                    │  FastAPI   :8000  ◀── /predict       │
                    │      │                               │
                    │      ▼                               │
                    │  MLflow    :5000  (model registry)   │
                    └──────────────────────────────────────┘
```

Three Docker containers managed by Docker Compose:

| Service | Image | Port | Purpose |
|---------|-------|------|---------|
| `mlflow` | `ghcr.io/mlflow/mlflow:v2.13.0` | 5000 | Model registry + experiment tracking |
| `api` | Built from `app/api/Dockerfile` | 8000 | FastAPI inference server |
| `frontend` | Built from `app/frontend/Dockerfile` | 8501 | Streamlit user interface |

Model load priority in the API:
1. MLflow Model Registry (`models:/CoveredCallLSTMCNN/Champion`) — if registry has the model
2. Local file fallback (`/app/saved_models/lstm_cnn_best_model.pth`) — always works

---

## 7. Quick Start — Run Locally

### Prerequisites
- Docker Desktop installed and running
- Git

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/wushuchris/covered-call-prototype.git
cd covered-call-prototype

# 2. Start all services
cd app
docker-compose up -d

# 3. Wait ~60 seconds for all containers to become healthy, then check:
docker-compose ps
```

**Access the services:**

| Service | URL |
|---------|-----|
| Streamlit UI | http://localhost:8501 |
| FastAPI docs | http://localhost:8000/docs |
| MLflow UI | http://localhost:5000 |

### Verify the API is up

```bash
curl http://localhost:8000/health
```

Expected response:
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

### Stop everything

```bash
docker-compose down
```

---

## 8. Deploy to AWS EC2

This deploys all three containers on a single EC2 `t3.medium` instance using Terraform.

### Prerequisites

- [Terraform](https://developer.hashicorp.com/terraform/install) >= 1.3 installed locally
- AWS CLI configured (`aws configure`) with your credentials
- An EC2 key pair created in the AWS Console

### Step 1 — Create an EC2 Key Pair

1. Go to **AWS Console → EC2 → Key Pairs → Create key pair**
2. Name it (e.g., `covered-call-key`), choose `.pem` format
3. Download the `.pem` file and move it to `~/.ssh/`
4. Set permissions: `chmod 400 ~/.ssh/covered-call-key.pem`

### Step 2 — Find Your IP Address

```bash
curl https://checkip.amazonaws.com
# Example output: 130.45.13.198
# Use as: 130.45.13.198/32
```

### Step 3 — Configure Terraform Variables

```bash
cd infra/terraform-app
cp terraform.tfvars.example terraform.tfvars
```

Edit `terraform.tfvars`:

```hcl
aws_region    = "us-east-1"
instance_type = "t3.medium"
key_pair_name = "covered-call-key"          # Name of your EC2 key pair (no .pem)
your_ip_cidr  = "130.45.13.198/32"          # Your IP from Step 2
repo_url      = "https://github.com/wushuchris/covered-call-prototype"
```

### Step 4 — Provision the EC2 Instance

```bash
terraform init
terraform plan    # Review what will be created
terraform apply   # Type 'yes' to confirm
```

Terraform will output:

```
instance_public_ip = "3.239.115.92"
streamlit_url      = "http://3.239.115.92:8501"
api_url            = "http://3.239.115.92:8000"
api_docs_url       = "http://3.239.115.92:8000/docs"
mlflow_url         = "http://3.239.115.92:5000"
ssh_command        = "ssh -i ~/.ssh/covered-call-key.pem ec2-user@3.239.115.92"
```

### Step 5 — Wait for Bootstrap to Complete

The EC2 instance automatically:
1. Installs Docker and Docker Compose
2. Clones this repository
3. Runs `docker-compose build && docker-compose up -d`

This takes **3–5 minutes** after `terraform apply` completes.

Check progress:
```bash
ssh -i ~/.ssh/covered-call-key.pem ec2-user@<your-ec2-ip>
tail -f /var/log/cloud-init-output.log
```

### Step 6 — Verify Deployment

```bash
# Check all containers are healthy
ssh -i ~/.ssh/covered-call-key.pem ec2-user@<ip>
cd /home/ec2-user/covered-call-prototype/app
docker-compose ps

# Check API health
curl http://<ip>:8000/health
```

All three containers should show `healthy`.

### Step 7 — Update After Code Changes

If you push code changes to GitHub:

```bash
ssh -i ~/.ssh/covered-call-key.pem ec2-user@<ip>
cd /home/ec2-user/covered-call-prototype
git pull origin main
cd app
docker-compose build api      # rebuild only the changed service
docker-compose up -d
```

### Tear Down

```bash
# From infra/terraform-app/
terraform destroy
```

---

## 9. API Reference

Base URL: `http://<host>:8000`

### `GET /health`
Liveness check. Returns model metadata.

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "ok",
  "device": "cpu",
  "model_source": "file",
  "model_uri": "/app/saved_models/lstm_cnn_best_model.pth",
  "seq_len": 50,
  "n_features": 35,
  "n_classes": 7
}
```

### `GET /model/info`
Returns full model metadata including feature names, class labels, and tuned thresholds.

```bash
curl http://localhost:8000/model/info
```

### `POST /predict`
Predict from a raw sequence array (JSON).

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sequences": [[[0.1, 0.2, ...], ...]]}'
```

Input shape: `[N, seq_len, n_features]` — e.g., `[1, 50, 35]`

```json
{
  "model_version": "lstm_cnn_v2_seq50",
  "model_source": "file",
  "n_predictions": 1,
  "results": [
    {
      "index": 0,
      "predicted_class": "OTM5_60_90",
      "class_index": 4,
      "probabilities": {
        "ATM_30": 0.05,
        "ATM_60": 0.08,
        "ATM_90": 0.07,
        "OTM5_30": 0.12,
        "OTM5_60_90": 0.42,
        "OTM10_30": 0.14,
        "OTM10_60_90": 0.12
      },
      "confidence": 0.42
    }
  ]
}
```

### `POST /predict/csv`
Upload a CSV or Parquet file and get predictions for all valid sequences.

**Required columns:** `symbol`, `date`, plus all model feature columns (see `/model/info`)

```bash
curl -X POST http://localhost:8000/predict/csv \
  -F "file=@your_stock_data.csv"
```

The API automatically groups by symbol, sorts by date, and builds sliding windows of `seq_len` rows per symbol.

### Interactive Docs

Full Swagger UI available at: `http://<host>:8000/docs`

---

## 10. MLflow Model Registry

### View the Registry

Open `http://localhost:5000` (or `http://<ec2-ip>:5000`) in your browser.

### Register the Best Model (from notebook)

Run `final_notebooks/10_mlflow_tracking.ipynb` with MLflow running.

The notebook:
1. Logs all model runs with hyperparameters and metrics
2. Registers the best LSTM-CNN as `CoveredCallLSTMCNN`
3. Sets the `Champion` alias on the best version

### Promote a New Model to Champion

In the MLflow UI:
1. Go to **Models → CoveredCallLSTMCNN**
2. Select the version you want to promote
3. Click **Add/Edit Aliases** → type `Champion` → Save

The API will automatically load the `Champion` version on next restart.

### Check Registered Models

```bash
curl http://localhost:5000/api/2.0/mlflow/registered-models/list
```

---

## 11. Environment Setup (Training)

For running notebooks locally:

```bash
# Create conda environment
conda env create -f environment.yaml
conda activate covered_call_ml

# Set up API keys
cp .env.example .env
# Edit .env and add:
#   ALPHA_VANTAGE_API_KEY=your_key
#   FRED_API_KEY=your_key

# Launch Jupyter
jupyter lab
```

### API Keys

| Key | Where to get |
|-----|-------------|
| `ALPHA_VANTAGE_API_KEY` | https://www.alphavantage.co/support/#api-key |
| `FRED_API_KEY` | https://fred.stlouisfed.org/docs/api/api_key.html |

### Download Pre-Built Datasets from S3

If you don't want to re-run data pulls from the APIs:

```bash
python data_scripts/data_dwnld.py
```

Downloads parquet files from `validex-ml-data` S3 bucket into `data/raw/`.

---

## 12. Feature Groups

| Group | Features |
|-------|----------|
| **Price & Returns** | Open, High, Low, Close, Volume, daily/weekly/monthly returns |
| **Technical** | Rolling volatility (5/20/60d), momentum, RSI, MACD, Bollinger Bands |
| **Valuation** | P/E ratio, P/S ratio, EV/EBITDA, Price/Book |
| **Profitability** | Gross/Operating/Net margin, ROA, ROE |
| **Leverage** | Debt/Equity ratio, Interest coverage, Current ratio |
| **Growth** | YoY revenue growth, EPS growth, FCF growth |
| **Macro (experimental)** | Fed funds rate, unemployment, yield curve spread, VIX |

Top 35 features selected via Random Forest feature importance (used by LSTM-CNN and PatchTST).

---

## 13. Target Variable

`optimal_bucket` — the covered call strategy that maximised risk-adjusted return on that day for that stock, defined as:

```
optimal_bucket = argmax over (moneyness × maturity) of risk_adjusted_return(t, t+expiry)
```

Where:
- **Moneyness**: ATM (delta 0.45–0.60), OTM5 (delta 0.30–0.45), OTM10 (delta 0.15–0.30)
- **Maturity**: 30 days, 60 days, 90 days

OTM5_60 and OTM5_90 were merged into `OTM5_60_90`; OTM10_60 and OTM10_90 into `OTM10_60_90` due to low frequency.

**Class distribution** is imbalanced (OTM strategies dominate in bull markets), handled via class weights and focal loss in the LSTM-CNN.

---

## 14. Data Splits

Time-based split (no look-ahead bias):

| Split | Date Range | Approx Rows |
|-------|-----------|-------------|
| Train | 2005–2021 | ~60,000 |
| Validation | 2022–2023 | ~5,000 |
| Test | 2024–2025 | ~2,500 |

StandardScaler fitted on training data only and applied to validation and test sets.

---

## License

See [LICENSE](LICENSE).
