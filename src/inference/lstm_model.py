"""
LSTM-CNN 7-class model module for covered call strategy.

Parallel to model.py (LGBM 3-class). Different features, different target,
different data granularity (daily vs monthly).

Loads the LSTM-CNN checkpoint from saved_models/, precomputes predictions
on all historical data, and caches to parquet. predict_bucket(ticker, date)
is an instant row lookup — same interface as the LGBM module.

Source data: daily_stock_optimal_bucket_modeling_with_fred.parquet
Model: lstm_cnn_best_model.pth (same as Streamlit deployment)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.preprocessing import StandardScaler

from src.utils import create_logger

logger = create_logger("lstm_model")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SAVED_MODELS_DIR = _PROJECT_ROOT / "saved_models"
FINAL_DATASETS_DIR = _PROJECT_ROOT / "final_datasets"
DATA_DIR = _PROJECT_ROOT / "data" / "processed"

FEATURE_STORE_PATH = DATA_DIR / "lstm_feature_store.parquet"
SEQ_LEN = 50

# 7-class labels (alphabetical — matches sklearn LabelEncoder from notebook 08)
CLASS_NAMES = [
    "ATM_30", "ATM_60", "ATM_90", "OTM10_30",
    "OTM10_60_90", "OTM5_30", "OTM5_60_90",
]

# 9→7 class merge (notebook 08 cell 7)
BUCKET_MAP = {
    "OTM10_60": "OTM10_60_90", "OTM10_90": "OTM10_60_90",
    "OTM5_60": "OTM5_60_90", "OTM5_90": "OTM5_60_90",
}

_model = None
_checkpoint = None
_feature_store = None


# ── Model architecture (from app/model/lstm_cnn.py) ─────────────────────

class _TemporalAttention(nn.Module):
    def __init__(self, hidden_dim, attn_dim=64):
        super().__init__()
        self.W = nn.Linear(hidden_dim, attn_dim, bias=True)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, lstm_out):
        weight = torch.softmax(self.v(torch.tanh(self.W(lstm_out))), dim=1)
        return (weight * lstm_out).sum(dim=1), weight.squeeze(-1)


class _LSTMCNNClassifier(nn.Module):
    def __init__(self, n_features, seq_len, num_classes,
                 cnn_out_channels=128, kernel_size=7,
                 lstm_hidden=128, lstm_layers=2,
                 attn_dim=128, dropout=0.2):
        super().__init__()
        pad = kernel_size // 2
        self.cnn = nn.Sequential(
            nn.Conv1d(n_features, cnn_out_channels, kernel_size, padding=pad),
            nn.BatchNorm1d(cnn_out_channels), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(cnn_out_channels, cnn_out_channels, kernel_size, padding=pad),
            nn.BatchNorm1d(cnn_out_channels), nn.ReLU(), nn.Dropout(dropout),
            nn.AdaptiveAvgPool1d(1),
        )
        self.lstm = nn.LSTM(
            n_features, lstm_hidden, lstm_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.attention = _TemporalAttention(lstm_hidden * 2, attn_dim)
        fusion = cnn_out_channels + lstm_hidden * 2
        self.head = nn.Sequential(
            nn.LayerNorm(fusion), nn.Dropout(dropout),
            nn.Linear(fusion, fusion // 2), nn.ReLU(),
            nn.Dropout(dropout / 2), nn.Linear(fusion // 2, num_classes),
        )

    def forward(self, x):
        cnn_out = self.cnn(x.transpose(1, 2)).squeeze(-1)
        lstm_out, _ = self.lstm(x)
        attn_out, _ = self.attention(lstm_out)
        return self.head(torch.cat([cnn_out, attn_out], dim=-1))


# ── Loading ──────────────────────────────────────────────────────────────

def _load_model():
    """Load LSTM-CNN checkpoint and instantiate model."""
    global _model, _checkpoint
    if _model is not None:
        return _model, _checkpoint

    ckpt = torch.load(
        SAVED_MODELS_DIR / "lstm_cnn_best_model.pth",
        map_location="cpu", weights_only=False,
    )
    p = ckpt["best_params"]
    model = _LSTMCNNClassifier(
        n_features=len(ckpt["feature_cols"]),
        seq_len=ckpt["seq_len"],
        num_classes=ckpt["num_classes"],
        cnn_out_channels=p.get("cnn_out_channels", 128),
        kernel_size=p.get("kernel_size", 7),
        lstm_hidden=p.get("lstm_hidden", 128),
        lstm_layers=p.get("lstm_layers", 2),
        attn_dim=p.get("attn_dim", 128),
        dropout=p.get("dropout", 0.2),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    _model, _checkpoint = model, ckpt
    logger.info("LSTM-CNN model loaded.")
    return model, ckpt


# ── Feature store builder ────────────────────────────────────────────────

def _build_feature_store():
    """Build precomputed LSTM predictions from the FRED daily dataset.

    Pipeline: load daily data → merge 9→7 classes → scale (fit on train) →
    build 50-day windows per symbol → batch inference → cache to parquet.
    """
    global _feature_store

    model, ckpt = _load_model()
    feature_cols = ckpt["feature_cols"]

    logger.info("Loading FRED dataset for LSTM feature store...")
    df = pd.read_parquet(FINAL_DATASETS_DIR / "daily_stock_optimal_bucket_modeling_with_fred.parquet")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

    # 9→7 class merge
    label_to_id = {name: i for i, name in enumerate(CLASS_NAMES)}
    df["label_7"] = df["optimal_bucket"].map(lambda x: BUCKET_MAP.get(x, x))
    df["label_id"] = df["label_7"].map(label_to_id)

    # Scale — fit on train only (< 2022), same as notebook 08
    scaler = StandardScaler()
    scaler.fit(df.loc[df["date"] < "2022-01-01", feature_cols])
    df[feature_cols] = scaler.transform(df[feature_cols])
    df[feature_cols] = df[feature_cols].fillna(0.0)

    # Build windows + run model per symbol
    records = []
    for symbol in sorted(df["symbol"].unique()):
        sym = df[df["symbol"] == symbol].sort_values("date").reset_index(drop=True)
        if len(sym) < SEQ_LEN:
            continue

        features = sym[feature_cols].values.astype(np.float32)
        n_win = len(features) - SEQ_LEN + 1
        windows = np.lib.stride_tricks.sliding_window_view(features, (SEQ_LEN, len(feature_cols)))
        windows = windows.squeeze(1)  # (n_win, SEQ_LEN, n_features)

        # Batch inference
        all_probs = []
        with torch.no_grad():
            for s in range(0, n_win, 128):
                batch = torch.from_numpy(windows[s:s + 128].copy())
                probs = torch.softmax(model(batch), dim=-1).numpy()
                all_probs.append(probs)

        probs_arr = np.vstack(all_probs)
        pred_ids = probs_arr.argmax(axis=1)

        # Each window's prediction date = last row of the window
        for i in range(n_win):
            row = sym.iloc[i + SEQ_LEN - 1]
            rec = {
                "symbol": symbol,
                "date": row["date"],
                "true_label": row["label_7"],
                "predicted_class_id": int(pred_ids[i]),
                "predicted_class": CLASS_NAMES[pred_ids[i]],
                "confidence": float(probs_arr[i].max()),
            }
            for j, cls in enumerate(CLASS_NAMES):
                rec[f"prob_{cls}"] = float(probs_arr[i, j])
            records.append(rec)

        logger.info(f"  {symbol}: {n_win} windows predicted")

    _feature_store = pd.DataFrame(records)
    _feature_store["date"] = pd.to_datetime(_feature_store["date"])
    _feature_store["year_month"] = _feature_store["date"].dt.to_period("M")
    _feature_store["correct"] = _feature_store["predicted_class"] == _feature_store["true_label"]
    _feature_store["sample_type"] = np.where(
        _feature_store["date"] < "2022-01-01", "Train Dataset",
        np.where(_feature_store["date"] < "2024-01-01", "Validation", "Test Dataset"),
    )

    _feature_store.to_parquet(FEATURE_STORE_PATH, index=False)
    logger.info(f"LSTM feature store saved: {len(_feature_store)} rows")


# ── Public API ───────────────────────────────────────────────────────────

def initialize():
    """Load or build the LSTM feature store. Safe to call multiple times."""
    global _feature_store
    if _feature_store is not None:
        return

    try:
        if FEATURE_STORE_PATH.exists():
            logger.info("Loading cached LSTM feature store...")
            _feature_store = pd.read_parquet(FEATURE_STORE_PATH)
            _feature_store["date"] = pd.to_datetime(_feature_store["date"])
            _feature_store["year_month"] = _feature_store["date"].dt.to_period("M")
            logger.info(f"LSTM feature store loaded: {len(_feature_store)} rows.")
        else:
            _build_feature_store()
        logger.info("LSTM model module initialized.")
    except Exception as e:
        logger.error(f"LSTM initialization failed: {e}")
        raise


def predict_bucket(ticker: str, date: str) -> dict:
    """Lookup LSTM-CNN prediction for a given ticker and date.

    Finds the closest available date in the feature store.

    Args:
        ticker: Stock symbol.
        date: Date string (YYYY-MM-DD).

    Returns:
        Dict with prediction, confidence, probabilities, or error.
    """
    global _feature_store
    if _feature_store is None:
        initialize()

    try:
        dt = pd.Timestamp(date)
        ticker_upper = ticker.upper()

        rows = _feature_store[_feature_store["symbol"] == ticker_upper]
        if rows.empty:
            return {"error": f"No LSTM data for {ticker_upper}"}

        # Nearest date lookup
        nearest_idx = (rows["date"] - dt).abs().idxmin()
        row = rows.loc[nearest_idx]
        snapped = row["date"].date() != dt.date()

        return {
            "ticker": ticker_upper,
            "date": date,
            "actual_date": str(row["date"].date()),
            "snapped": snapped,
            "model_name": "LSTM-CNN 7-Class",
            "predicted_class": row["predicted_class"],
            "confidence": round(float(row["confidence"]), 4),
            "true_label": row["true_label"],
            "correct": bool(row["correct"]),
            "sample_type": row["sample_type"],
            "probabilities": {
                cls: round(float(row[f"prob_{cls}"]), 4)
                for cls in CLASS_NAMES
            },
        }

    except Exception as e:
        logger.error(f"LSTM predict_bucket failed: {e}")
        return {"error": f"LSTM prediction failed: {e}"}
