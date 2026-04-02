"""
FastAPI inference server for the LSTM-CNN Covered Call Strategy Classifier.

Endpoints:
  GET  /health          → liveness check
  GET  /model/info      → model metadata (features, classes, params)
  POST /predict         → predict from raw sequence array
  POST /predict/csv     → upload a CSV file and predict
"""

import os
import io
import sys
import logging
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add app root to path so we can import the model module
sys.path.append(str(Path(__file__).resolve().parents[1]))
from model.lstm_cnn import load_model, predict

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Config from environment ────────────────────────────────────────────────
MODEL_PATH = os.getenv("MODEL_PATH", "/app/saved_models/lstm_cnn_best_model.pth")
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Load model at startup ──────────────────────────────────────────────────
logger.info(f"Loading model from {MODEL_PATH} on {DEVICE}")
model, ckpt = load_model(MODEL_PATH, DEVICE)

SEQ_LEN       = ckpt["seq_len"]
FEATURE_COLS  = ckpt["feature_cols"]
TARGET_CLASSES = ckpt["target_classes"]
NUM_CLASSES   = ckpt["num_classes"]
THRESHOLDS    = np.array(ckpt.get("tuned_thresholds", [1.0] * NUM_CLASSES))

logger.info(f"Model loaded — {NUM_CLASSES} classes, seq_len={SEQ_LEN}, "
            f"n_features={len(FEATURE_COLS)}")

# ── FastAPI app ────────────────────────────────────────────────────────────
app = FastAPI(
    title="Covered Call Strategy Classifier",
    description="LSTM-CNN + Attention model for classifying optimal covered call strategy buckets",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ────────────────────────────────────────────────────────────────
class SequenceRequest(BaseModel):
    """
    Raw sequence input.
    sequences: list of N sequences, each of shape (seq_len, n_features).
    """
    sequences: List[List[List[float]]]


class PredictionResult(BaseModel):
    index: int
    predicted_class: str
    class_index: int
    probabilities: dict   # {class_name: probability}
    confidence: float


class PredictionResponse(BaseModel):
    model_version: str
    n_predictions: int
    results: List[PredictionResult]


# ── Helpers ────────────────────────────────────────────────────────────────
def build_response(preds, probs_arr) -> PredictionResponse:
    results = []
    for i, (pred_idx, prob_row) in enumerate(zip(preds, probs_arr)):
        prob_dict = {cls: round(float(p), 4)
                     for cls, p in zip(TARGET_CLASSES, prob_row)}
        results.append(PredictionResult(
            index           = i,
            predicted_class = TARGET_CLASSES[pred_idx],
            class_index     = int(pred_idx),
            probabilities   = prob_dict,
            confidence      = round(float(prob_row[pred_idx]), 4),
        ))
    return PredictionResponse(
        model_version  = f"lstm_cnn_v1_seq{SEQ_LEN}",
        n_predictions  = len(results),
        results        = results,
    )


def df_to_sequences(df: pd.DataFrame) -> np.ndarray:
    """
    Convert a flat DataFrame (rows = timesteps across symbols/dates)
    to sequences of shape (N, SEQ_LEN, n_features).

    Expects columns: symbol, date, + all feature columns.
    Rows must be sorted by symbol then date (ascending).
    """
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"CSV missing required feature columns: {missing[:5]}{'...' if len(missing)>5 else ''}"
        )

    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    X_seq = []
    for sym, grp in df.groupby("symbol"):
        grp    = grp.reset_index(drop=True)
        X_vals = grp[FEATURE_COLS].fillna(0).values.astype(np.float32)
        if len(grp) < SEQ_LEN:
            continue
        for i in range(SEQ_LEN - 1, len(grp)):
            X_seq.append(X_vals[i - SEQ_LEN + 1: i + 1])
    if not X_seq:
        raise HTTPException(
            status_code=422,
            detail=f"No valid sequences found. Each symbol needs >= {SEQ_LEN} rows."
        )
    return np.array(X_seq, dtype=np.float32)


# ── Endpoints ──────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status"    : "ok",
        "device"    : str(DEVICE),
        "model_path": MODEL_PATH,
        "seq_len"   : SEQ_LEN,
        "n_features": len(FEATURE_COLS),
        "n_classes" : NUM_CLASSES,
    }


@app.get("/model/info")
def model_info():
    return {
        "seq_len"       : SEQ_LEN,
        "n_features"    : len(FEATURE_COLS),
        "feature_cols"  : FEATURE_COLS,
        "target_classes": TARGET_CLASSES,
        "best_params"   : ckpt["best_params"],
        "thresholds"    : {cls: round(float(t), 4)
                           for cls, t in zip(TARGET_CLASSES, THRESHOLDS)},
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_sequences(request: SequenceRequest):
    """
    Predict from raw sequences.
    Input shape per sequence: (seq_len=50, n_features=35)
    """
    X = np.array(request.sequences, dtype=np.float32)
    if X.ndim != 3:
        raise HTTPException(status_code=422, detail="sequences must be 3-D: [N, seq_len, n_features]")
    if X.shape[1] != SEQ_LEN:
        raise HTTPException(status_code=422,
                            detail=f"seq_len mismatch: expected {SEQ_LEN}, got {X.shape[1]}")
    if X.shape[2] != len(FEATURE_COLS):
        raise HTTPException(status_code=422,
                            detail=f"n_features mismatch: expected {len(FEATURE_COLS)}, got {X.shape[2]}")

    preds, probs = predict(model, X, THRESHOLDS, DEVICE)
    return build_response(preds, probs)


@app.post("/predict/csv", response_model=PredictionResponse)
async def predict_from_csv(file: UploadFile = File(...)):
    """
    Upload a CSV (or parquet) file with columns: symbol, date, + feature columns.
    The API builds rolling sequences per symbol and returns predictions.
    """
    content = await file.read()
    try:
        if file.filename.endswith(".parquet"):
            df = pd.read_parquet(io.BytesIO(content))
        else:
            df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not parse file: {e}")

    if "date" not in df.columns or "symbol" not in df.columns:
        raise HTTPException(status_code=422,
                            detail="File must contain 'symbol' and 'date' columns.")

    df["date"] = pd.to_datetime(df["date"])
    X          = df_to_sequences(df)
    preds, probs = predict(model, X, THRESHOLDS, DEVICE)
    return build_response(preds, probs)
