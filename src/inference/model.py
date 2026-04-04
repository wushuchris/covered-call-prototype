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

    # ── Use model's actual trained feature names (not stale metadata) ──
    # Metadata lists 27 pre-IV features, but the model was trained on 34
    # (after dropping 5 leaky features and adding 9 IV features).
    trained_features = model.feature_name_
    logger.info(f"Model trained on {len(trained_features)} features: {trained_features[:5]}...")

    # ── Ensure all model features exist and handle NaN ──
    for col in trained_features:
        if col not in mdata.columns:
            mdata[col] = 0.0
    mdata[trained_features] = mdata[trained_features].fillna(mdata[trained_features].median())

    # ── Run model on all rows ──
    logger.info(f"Running model predictions with {len(trained_features)} features...")
    X = mdata[trained_features].values
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)

    mdata["model_moneyness_id"] = predictions
    mdata["model_moneyness"] = [MONEYNESS_MAP[p] for p in predictions]
    mdata["model_confidence"] = probabilities.max(axis=1)

    # Store full probability distribution for risk-adjusted strategy
    mdata["prob_ATM"] = probabilities[:, 0]
    mdata["prob_OTM5"] = probabilities[:, 1]
    mdata["prob_OTM10"] = probabilities[:, 2]

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
            "sample_type": "Test Dataset" if actual_month >= OOS_CUTOFF else "Train Dataset",
        }

    except Exception as e:
        logger.error(f"predict_bucket failed: {e}")
        return {"error": f"Prediction failed: {e}", "ticker": ticker, "date": date}


def compute_model_metrics(year: str = "all", sample_type: str = "all") -> dict:
    """Compute model performance metrics from the feature store.

    Calculates accuracy, macro F1, top-2 accuracy, and per-class
    precision/recall/F1 from precomputed predictions vs ground truth.

    Args:
        year: Filter by year ('all' or e.g. '2020').
        sample_type: Filter by sample type ('all', 'train', 'test').

    Returns:
        Dict with summary metrics, per-class breakdown, per-year breakdown,
        and confidence analysis.
    """
    global _feature_store

    if _feature_store is None:
        initialize()

    try:
        df = _feature_store.copy()

        # Apply filters
        if year != "all":
            df = df[df["decision_date"].dt.year == int(year)]
        if sample_type == "train":
            df = df[df["year_month"] < OOS_CUTOFF]
        elif sample_type == "test":
            df = df[df["year_month"] >= OOS_CUTOFF]

        if df.empty:
            return {"error": f"No data for year={year}, sample={sample_type}"}

        n = len(df)
        accuracy = float(df["model_correct"].mean())
        top2 = float(df["model_top2_hit"].mean())

        # Per-class metrics
        classes = ["ATM", "OTM5", "OTM10"]
        per_class = {}
        for cls in classes:
            true_mask = df["true_moneyness"] == cls
            pred_mask = df["model_moneyness"] == cls
            tp = int((true_mask & pred_mask).sum())
            fp = int((~true_mask & pred_mask).sum())
            fn = int((true_mask & ~pred_mask).sum())
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            per_class[cls] = {
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "support": int(true_mask.sum()),
                "predicted": int(pred_mask.sum()),
            }

        # Macro F1
        macro_f1 = sum(v["f1"] for v in per_class.values()) / len(classes)

        # Per-year breakdown
        df["year"] = df["decision_date"].dt.year
        per_year = []
        for yr, grp in df.groupby("year"):
            yr_acc = float(grp["model_correct"].mean())
            yr_top2 = float(grp["model_top2_hit"].mean())
            yr_n = len(grp)
            # Per-year macro F1
            yr_f1s = []
            for cls in classes:
                t = grp["true_moneyness"] == cls
                p = grp["model_moneyness"] == cls
                tp = int((t & p).sum())
                fp = int((~t & p).sum())
                fn = int((t & ~p).sum())
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                yr_f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0)
            per_year.append({
                "year": int(yr),
                "accuracy": round(yr_acc, 4),
                "macro_f1": round(sum(yr_f1s) / len(yr_f1s), 4),
                "top2": round(yr_top2, 4),
                "n_samples": yr_n,
            })

        # Confidence analysis
        correct = df[df["model_correct"]]
        incorrect = df[~df["model_correct"]]
        conf_analysis = {
            "avg_when_correct": round(float(correct["model_confidence"].mean()), 4) if len(correct) > 0 else 0.0,
            "avg_when_incorrect": round(float(incorrect["model_confidence"].mean()), 4) if len(incorrect) > 0 else 0.0,
            "overall_avg": round(float(df["model_confidence"].mean()), 4),
        }

        return {
            "n_samples": n,
            "year_filter": year,
            "sample_filter": sample_type,
            "accuracy": round(accuracy, 4),
            "macro_f1": round(macro_f1, 4),
            "top2_accuracy": round(top2, 4),
            "per_class": per_class,
            "per_year": per_year,
            "confidence": conf_analysis,
            "model_name": "LGBM 3-Class Moneyness (Walk-Forward)",
        }

    except Exception as e:
        logger.error(f"compute_model_metrics failed: {e}")
        return {"error": f"Metrics computation failed: {e}"}


def get_lgbm_experiment_info() -> dict:
    """Return LGBM production model info formatted like an MLflow run.

    Computes test metrics from the feature store and extracts
    hyperparameters from the model object.

    Returns:
        Dict matching the mlflow_reader run format.
    """
    global _feature_store

    if _feature_store is None:
        initialize()

    try:
        from sklearn.metrics import balanced_accuracy_score, f1_score, accuracy_score

        model, _ = _load_model()

        # Test set metrics (year_month >= 2025-01)
        test = _feature_store[_feature_store["year_month"] >= OOS_CUTOFF]
        y_true = test["true_moneyness"].map(MONEYNESS_TO_ID)
        y_pred = test["model_moneyness_id"]

        test_acc = float(accuracy_score(y_true, y_pred))
        test_f1 = float(f1_score(y_true, y_pred, average="macro"))
        test_bal = float(balanced_accuracy_score(y_true, y_pred))

        # Walk-forward val F1 = 0.47 from CLAUDE.md (not recomputable here)
        val_f1 = 0.4682

        # Hyperparameters
        raw_params = model.get_params()
        params = {
            k: str(v) for k, v in sorted(raw_params.items())
            if v is not None and str(v) not in ("-1", "None")
        }

        return {
            "run_id": "lgbm-production",
            "run_name": "LGBM 3-Class (Production)",
            "experiment_id": "local",
            "model_type": "LightGBM Walk-Forward",
            "variant": "production",
            "n_classes": "3",
            "n_features": str(len(model.feature_name_)),
            "seq_len": "",
            "metrics": {
                "val_macro_f1": round(val_f1, 4),
                "test_macro_f1": round(test_f1, 4),
                "test_accuracy": round(test_acc, 4),
                "test_balanced_accuracy": round(test_bal, 4),
            },
            "params": params,
            "artifacts": {
                "confusion_matrix": None,
                "roc_curves": None,
            },
        }

    except Exception as e:
        logger.error(f"get_lgbm_experiment_info failed: {e}")
        return {}


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
