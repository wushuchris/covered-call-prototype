"""
Core model module for the covered call strategy.

Loads the LGBM 3-class moneyness model and precomputes a feature store
at startup from historical parquets. predict_bucket(ticker, date) is a
row lookup returning model prediction + ground truth comparison.

Startup flow:
    initialize() → _load_model() + _build_feature_store()

Inference flow:
    predict_bucket(ticker, date) → snap to month → row lookup → dict
"""

import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from functools import lru_cache

from src.utils import create_logger

logger = create_logger("model")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = _PROJECT_ROOT / "models"
DATA_DIR = _PROJECT_ROOT / "data" / "processed"

UNIVERSE = ["AAPL", "AMZN", "AVGO", "GOOG", "GOOGL", "META", "MSFT", "NVDA", "TSLA", "WMT"]

MONEYNESS_MAP = {0: "ATM", 1: "OTM5", 2: "OTM10"}
MONEYNESS_TO_ID = {"ATM": 0, "OTM5": 1, "OTM10": 2}

# The saved model was trained on data['year'] < 2025 (notebook 06c cell 21)
OOS_CUTOFF = pd.Period("2025-01", freq="M")

# Features the model was trained without (dropped in 06b/06c)
DROP_FEATURES = ["adjusted_close", "volume", "operating_margin", "price_to_sma50", "vol_10d"]

# Feature store cache path — saved to disk so teammates can inspect/replicate
FEATURE_STORE_PATH = DATA_DIR / "feature_store.parquet"

# ── Singleton state ──────────────────────────────────────────────────────────
_model = None
_metadata = None
_feature_store = None  # pd.DataFrame — loaded from disk or built once


# ── Model loading ────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_model():
    """Load LGBM 3-class moneyness model and metadata from disk.

    Returns:
        Tuple of (model, metadata dict).
    """
    try:
        model = joblib.load(MODELS_DIR / "lgbm_3class_moneyness.joblib")
        with open(MODELS_DIR / "improved_model_metadata.json") as f:
            metadata = json.load(f)
        logger.info("LGBM 3-class moneyness model loaded.")
        return model, metadata
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


# ── IV feature computation ───────────────────────────────────────────────────

def _compute_iv_features(options_df):
    """Compute monthly IV features per ticker from call options.

    Replicates the IV feature engineering from notebook 06c.
    Uses ATM calls (delta 0.35-0.65) for stable IV estimates.

    Args:
        options_df: Cleaned options DataFrame (all puts + calls).

    Returns:
        DataFrame with columns [symbol, year_month, iv_mean, iv_median,
        iv_skew, iv_short_mean, iv_short_std, iv_long_mean,
        iv_term_structure, iv_rank, iv_change].
    """
    try:
        calls = options_df[options_df["call_put"] == "CALL"].copy()
        calls["trade_date"] = pd.to_datetime(calls["trade_date"])
        calls["expiration"] = pd.to_datetime(calls["expiration"])
        calls["dte"] = (calls["expiration"] - calls["trade_date"]).dt.days
        calls["year_month"] = calls["trade_date"].dt.to_period("M")

        # ATM calls only
        atm = calls[(calls["delta"] >= 0.35) & (calls["delta"] <= 0.65)]

        # Short-term IV (DTE <= 45)
        short_iv = (
            atm[atm["dte"] <= 45]
            .groupby(["symbol", "year_month"])
            .agg(iv_short_mean=("implied_vol", "mean"), iv_short_std=("implied_vol", "std"))
            .reset_index()
        )

        # Long-term IV (DTE > 45)
        long_iv = (
            atm[atm["dte"] > 45]
            .groupby(["symbol", "year_month"])
            .agg(iv_long_mean=("implied_vol", "mean"))
            .reset_index()
        )

        # Overall IV stats per (ticker, month)
        overall = (
            atm.groupby(["symbol", "year_month"])
            .agg(
                iv_mean=("implied_vol", "mean"),
                iv_median=("implied_vol", "median"),
                iv_skew=("implied_vol", "skew"),
            )
            .reset_index()
        )

        # Merge short + long + overall
        iv = overall.merge(short_iv, on=["symbol", "year_month"], how="left")
        iv = iv.merge(long_iv, on=["symbol", "year_month"], how="left")

        # Derived: term structure
        iv["iv_term_structure"] = iv["iv_long_mean"] - iv["iv_short_mean"]

        # Derived: IV rank (12-month percentile)
        iv = iv.sort_values(["symbol", "year_month"])
        rolling_high = iv.groupby("symbol")["iv_mean"].transform(
            lambda x: x.rolling(12, min_periods=1).max()
        )
        rolling_low = iv.groupby("symbol")["iv_mean"].transform(
            lambda x: x.rolling(12, min_periods=1).min()
        )
        iv_range = (rolling_high - rolling_low).replace(0, np.nan)
        iv["iv_rank"] = (iv["iv_mean"] - rolling_low) / iv_range

        # Derived: IV change (month-over-month)
        iv["iv_change"] = iv.groupby("symbol")["iv_mean"].pct_change()

        logger.info(f"IV features computed: {len(iv)} rows.")
        return iv

    except Exception as e:
        logger.error(f"IV feature computation failed: {e}")
        raise


# ── Feature store builder ────────────────────────────────────────────────────

def _build_feature_store():
    """Build the precomputed feature store from historical parquets.

    Loads modeling_data.parquet (features + ground truth labels),
    computes IV features from options, merges them, runs the LGBM
    model on every row, and stores model predictions alongside
    ground truth for instant lookup.

    Sets the module-level _feature_store DataFrame.
    """
    global _feature_store

    model, metadata = _load_model()
    feature_cols = metadata["feature_cols"]

    # ── Load base modeling data (features + labels from notebook 05) ──
    logger.info("Loading modeling data...")
    mdata = pd.read_parquet(DATA_DIR / "modeling_data.parquet")
    mdata["decision_date"] = pd.to_datetime(mdata["decision_date"])
    mdata["year_month"] = mdata["decision_date"].dt.to_period("M")

    # ── Load options and compute IV features ──
    logger.info("Loading options for IV computation...")
    options = pd.read_parquet(DATA_DIR / "options_clean.parquet")
    options = options[options["symbol"].isin(UNIVERSE)]
    iv = _compute_iv_features(options)

    # Free options memory — not needed after IV computation
    del options

    # ── Merge IV features into modeling data ──
    iv_cols = [
        "iv_mean", "iv_median", "iv_skew", "iv_short_mean", "iv_short_std",
        "iv_long_mean", "iv_term_structure", "iv_rank", "iv_change",
    ]
    mdata = mdata.merge(
        iv[["symbol", "year_month"] + iv_cols],
        on=["symbol", "year_month"],
        how="left",
    )

    # Forward-fill IV per ticker, then fill remaining with median
    mdata = mdata.sort_values(["symbol", "decision_date"])
    mdata[iv_cols] = mdata.groupby("symbol")[iv_cols].ffill()
    mdata[iv_cols] = mdata[iv_cols].fillna(mdata[iv_cols].median())

    # ── Ensure all model features exist and handle NaN ──
    for col in feature_cols:
        if col not in mdata.columns:
            mdata[col] = 0.0
    mdata[feature_cols] = mdata[feature_cols].fillna(mdata[feature_cols].median())

    # ── Run model on all rows ──
    logger.info("Running model predictions...")
    X = mdata[feature_cols].values
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)

    mdata["model_moneyness_id"] = predictions
    mdata["model_moneyness"] = [MONEYNESS_MAP[p] for p in predictions]
    mdata["model_confidence"] = probabilities.max(axis=1)

    # ── Apply maturity rule: iv_rank > 0.5 → SHORT, else LONG ──
    mdata["model_maturity"] = np.where(
        mdata["iv_rank"].fillna(0) > 0.5, "SHORT", "LONG"
    )
    mdata["model_bucket"] = mdata["model_moneyness"] + "_" + mdata["model_maturity"]

    # ── Ground truth comparison ──
    mdata["true_moneyness"] = mdata["best_bucket"].str.extract(r"(ATM|OTM5|OTM10)")[0]
    mdata["model_correct"] = mdata["model_moneyness"] == mdata["true_moneyness"]

    # Top-2 accuracy
    top2_classes = np.argsort(probabilities, axis=1)[:, -2:]
    true_ids = mdata["true_moneyness"].map(MONEYNESS_TO_ID).values
    mdata["model_top2_hit"] = [
        true_ids[i] in top2_classes[i] for i in range(len(true_ids))
    ]

    # ── Save to disk ──
    mdata.to_parquet(FEATURE_STORE_PATH, index=False)
    logger.info(f"Feature store saved to {FEATURE_STORE_PATH}")

    _feature_store = mdata
    logger.info(
        f"Feature store ready: {len(_feature_store)} rows, "
        f"{mdata['symbol'].nunique()} tickers, "
        f"{mdata['year_month'].min()} to {mdata['year_month'].max()}"
    )


# ── Public API ───────────────────────────────────────────────────────────────

def initialize():
    """Called at server startup. Loads model and builds feature store.

    Tries to load from disk first (instant). If no cached file exists,
    builds from raw parquets and saves to disk for next time.

    Safe to call multiple times — skips if already initialized.
    """
    global _feature_store
    if _feature_store is not None:
        return

    try:
        if FEATURE_STORE_PATH.exists():
            logger.info(f"Loading feature store from {FEATURE_STORE_PATH}...")
            _feature_store = pd.read_parquet(FEATURE_STORE_PATH)
            _feature_store["decision_date"] = pd.to_datetime(_feature_store["decision_date"])
            _feature_store["year_month"] = _feature_store["decision_date"].dt.to_period("M")
            _load_model()  # still need model in memory for get_date_range etc.
            logger.info(f"Feature store loaded: {len(_feature_store)} rows.")
        else:
            logger.info("No cached feature store found — building from raw data...")
            _build_feature_store()

        logger.info("Model module initialized successfully.")
    except Exception as e:
        logger.error(f"Model initialization failed: {e}")
        raise


def predict_bucket(ticker: str, date: str) -> dict:
    """Lookup prediction for a given ticker and date.

    Snaps the date to its calendar month and returns the model's
    prediction alongside the ground truth best bucket.

    Args:
        ticker: Stock symbol (e.g. "AAPL").
        date: Any date string parseable by pandas (e.g. "2021-09-14").

    Returns:
        Dict with model prediction, ground truth, and comparison.
    """
    global _feature_store

    if _feature_store is None:
        initialize()

    try:
        dt = pd.Timestamp(date)
        year_month = dt.to_period("M")
        ticker_upper = ticker.upper()

        ticker_rows = _feature_store[_feature_store["symbol"] == ticker_upper]

        if ticker_rows.empty:
            return {
                "error": f"No data for {ticker_upper}",
                "ticker": ticker_upper,
                "date": date,
            }

        # Exact month match first, then snap to nearest available month
        matches = ticker_rows[ticker_rows["year_month"] == year_month]
        snapped = False

        if matches.empty:
            available = ticker_rows["year_month"].unique()
            diffs = [abs((p - year_month).n) for p in available]
            nearest = available[pd.Series(diffs).argmin()]
            matches = ticker_rows[ticker_rows["year_month"] == nearest]
            snapped = True
            logger.info(f"Snapped {ticker_upper} {year_month} → {nearest}")

        row = matches.iloc[0]
        actual_month = row["year_month"]

        return {
            "ticker": ticker_upper,
            "date": date,
            "month": str(actual_month),
            "snapped": snapped,
            # Model prediction
            "model_bucket": row["model_bucket"],
            "model_moneyness": row["model_moneyness"],
            "model_maturity": row["model_maturity"],
            "model_confidence": round(float(row["model_confidence"]), 4),
            # Ground truth
            "best_bucket": row["best_bucket"],
            "best_return": round(float(row["best_return"]), 4),
            # Comparison
            "model_correct": bool(row["model_correct"]),
            "model_top2_hit": bool(row["model_top2_hit"]),
            # Baseline
            "baseline": "OTM10_SHORT",
            "status": "historical",
            "sample_type": "out-of-sample" if actual_month >= OOS_CUTOFF else "in-sample",
        }

    except Exception as e:
        logger.error(f"predict_bucket failed: {e}")
        return {"error": f"Prediction failed: {e}", "ticker": ticker, "date": date}


def get_date_range() -> dict:
    """Return the valid date range for predictions.

    Returns:
        Dict with min_date, max_date, and available tickers.
    """
    global _feature_store

    if _feature_store is None:
        initialize()

    return {
        "min_date": str(_feature_store["decision_date"].min().date()),
        "max_date": str(_feature_store["decision_date"].max().date()),
        "tickers": sorted(_feature_store["symbol"].unique().tolist()),
        "total_rows": len(_feature_store),
    }
