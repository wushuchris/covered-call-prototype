# Backtesting pipeline for the covered call strategy.
#
# Loops through historical months, applies scoring + allocation each month,
# computes realized returns, and produces a report with metrics for the
# strategy and both baselines.
#
# Cache-first: if a report exists on disk for the given preset, return it.
# Otherwise run the full backtest and cache the result.
#
# The backtest loop runs in plain Python using the scoring engine directly.

import json
import numpy as np
import pandas as pd
from pathlib import Path


from src.inference.scoring import (
    score_month, allocate, compute_metrics,
    # Baselines deferred — pending per-bucket realized returns
    # baseline_no_model, baseline_no_scoring,
)
from src.utils import create_logger

logger = create_logger("backtesting")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = _PROJECT_ROOT / "data" / "processed"
CACHE_DIR = _PROJECT_ROOT / "cache"


def _cache_path(preset: str) -> Path:
    """Cache file path for a given preset.

    Args:
        preset: Strategy preset name.

    Returns:
        Path to the cached JSON report.
    """
    return CACHE_DIR / f"backtest_{preset}.json"


def _load_cached_report(preset: str) -> dict | None:
    """Check if a cached backtest report exists for this preset.

    Args:
        preset: Strategy preset name.

    Returns:
        Cached report dict, or None if no cache found.
    """
    try:
        path = _cache_path(preset)
        if path.exists():
            return json.loads(path.read_text())
    except Exception:
        pass
    return None


def _save_cached_report(report: dict, preset: str) -> None:
    """Persist a backtest report to disk.

    Args:
        report: Report dict to save.
        preset: Strategy preset name.
    """
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _cache_path(preset).write_text(json.dumps(report, indent=2, default=str))
    except Exception:
        pass


def _compute_argmax_return(month_data: pd.DataFrame,
                           bucket_returns: pd.DataFrame, period) -> float:
    """Compute return for Argmax strategy: model's top pick, all tickers, equal weight.

    Args:
        month_data: Feature store rows for one month (all tickers).
        bucket_returns: Per-bucket returns DataFrame.
        period: The year_month Period for this month.

    Returns:
        Equally-weighted average return across all tradeable tickers.
    """
    try:
        month_br = bucket_returns[bucket_returns["year_month"] == period]
        returns = []
        for _, row in month_data.iterrows():
            moneyness = row["model_moneyness"]
            col = f"return_{moneyness}"
            ticker_row = month_br[month_br["symbol"] == row["symbol"]]
            if ticker_row.empty or col not in ticker_row.columns:
                continue
            val = ticker_row[col].iloc[0]
            if not pd.isna(val):
                returns.append(float(val))
        return float(np.mean(returns)) if returns else 0.0
    except Exception:
        return 0.0


def _compute_risk_adjusted_return(month_data: pd.DataFrame,
                                  bucket_returns: pd.DataFrame,
                                  period, avg_returns: dict) -> float:
    """Compute return for Risk-Adjusted strategy: probability × expected return.

    For each ticker, picks the bucket that maximizes P(bucket) × E[return|bucket],
    then equal-weights across all tickers.

    Args:
        month_data: Feature store rows for one month (all tickers).
        bucket_returns: Per-bucket returns DataFrame.
        period: The year_month Period for this month.
        avg_returns: Dict of average historical returns per bucket {ATM: x, OTM5: y, OTM10: z}.

    Returns:
        Equally-weighted average return across all tradeable tickers.
    """
    try:
        month_br = bucket_returns[bucket_returns["year_month"] == period]
        returns = []
        for _, row in month_data.iterrows():
            # Score each bucket: P(bucket) × E[return|bucket]
            scores = {}
            for cls in ["ATM", "OTM5", "OTM10"]:
                prob = row.get(f"prob_{cls}", 0.0)
                expected = avg_returns.get(cls, 0.0)
                scores[cls] = prob * expected

            # Pick the bucket with the highest risk-adjusted score
            best_bucket = max(scores, key=scores.get)
            col = f"return_{best_bucket}"

            ticker_row = month_br[month_br["symbol"] == row["symbol"]]
            if ticker_row.empty or col not in ticker_row.columns:
                continue
            val = ticker_row[col].iloc[0]
            if not pd.isna(val):
                returns.append(float(val))
        return float(np.mean(returns)) if returns else 0.0
    except Exception:
        return 0.0


def _compute_monthly_return(allocated: pd.DataFrame, budget: float,
                            bucket_returns: pd.DataFrame, period) -> float:
    """Compute weighted portfolio return for a single month.

    Looks up the realized covered call return for each ticker's
    chosen moneyness bucket from bucket_returns.parquet (notebook 05b).

    Args:
        allocated: DataFrame with 'allocation' and 'model_bucket' columns.
        budget: Total budget for normalization.
        bucket_returns: Per-bucket returns (symbol, year_month, return_ATM/OTM5/OTM10).
        period: The year_month Period for this month.

    Returns:
        Portfolio return as a float.
    """
    try:
        if allocated.empty or budget <= 0:
            return 0.0

        month_br = bucket_returns[bucket_returns["year_month"] == period]
        tradeable = []

        for _, row in allocated.iterrows():
            ticker = row["symbol"]
            bucket = row["model_bucket"]
            col = f"return_{bucket}"

            ticker_row = month_br[month_br["symbol"] == ticker]
            if ticker_row.empty or col not in ticker_row.columns:
                continue
            val = ticker_row[col].iloc[0]
            if pd.isna(val):
                continue
            tradeable.append((row["allocation"], float(val)))

        if not tradeable:
            return 0.0

        # Re-weight: redistribute budget only across tradeable positions
        allocs, rets = zip(*tradeable)
        total_alloc = sum(allocs)
        weights = np.array(allocs) / total_alloc if total_alloc > 0 else np.zeros(len(allocs))
        return float(np.dot(weights, np.array(rets)))

    except Exception:
        return 0.0


def _run_backtest_loop(feature_store: pd.DataFrame, budget: float,
                       preset: str) -> dict:
    """Run the full backtest loop across all months.

    For each month:
        1. Score and allocate (strategy)
        2. Look up per-bucket realized returns
        3. Compute weighted portfolio return

    Args:
        feature_store: Full feature store DataFrame.
        budget: Dollar budget per month.
        preset: Strategy preset name.

    Returns:
        Dict with monthly returns and computed metrics.
    """
    # Load per-bucket returns (from notebook 05b, stored under src/data)
    # NOTE: currently under src/data — may move to data/processed later
    br_path = _PROJECT_ROOT / "src" / "data" / "bucket_returns.parquet"
    if br_path.exists():
        bucket_returns = pd.read_parquet(br_path)
        bucket_returns["year_month"] = bucket_returns["year_month"].astype("period[M]")
        logger.info(f"Loaded bucket returns: {len(bucket_returns)} rows")
    else:
        logger.warning("bucket_returns.parquet not found — returns will be zero")
        bucket_returns = pd.DataFrame()

    months = sorted(feature_store["year_month"].unique())
    prev_buckets = {}

    strategy_returns = []
    baseline_returns = []
    argmax_returns = []
    risk_adj_returns = []

    # Compute average historical returns per bucket for risk-adjusted scoring
    # Uses expanding window: for each month, only use data available up to that point
    avg_returns = {"ATM": 0.0, "OTM5": 0.0, "OTM10": 0.0}
    cumulative_sums = {"ATM": 0.0, "OTM5": 0.0, "OTM10": 0.0}
    cumulative_counts = {"ATM": 0, "OTM5": 0, "OTM10": 0}

    # Per-month detail for the UI timeline
    monthly_detail = []

    for i, month in enumerate(months):
        month_data = feature_store[feature_store["year_month"] == month].copy()

        if month_data.empty:
            continue

        # ── Strategy: score → allocate → return ──
        try:
            scored = score_month(month_data, prev_buckets, preset=preset)
            allocated = allocate(scored, budget, preset=preset)
            strat_return = _compute_monthly_return(allocated, budget, bucket_returns, month)
        except Exception as e:
            logger.error(f"Strategy failed for {month}: {e}")
            strat_return = 0.0
            allocated = pd.DataFrame()

        # ── Baseline: OTM10 on all tickers, equal weight ──
        month_br = bucket_returns[bucket_returns["year_month"] == month]
        tickers_with_data = month_br[month_br["return_OTM10"].notna()]
        if not tickers_with_data.empty:
            baseline_ret = float(tickers_with_data["return_OTM10"].mean())
        else:
            baseline_ret = 0.0

        # ── Argmax: model's top pick, all tickers, equal weight ──
        argmax_ret = _compute_argmax_return(month_data, bucket_returns, month)

        # ── Risk-Adjusted: P(bucket) × E[return|bucket], best score wins ──
        risk_adj_ret = _compute_risk_adjusted_return(
            month_data, bucket_returns, month, avg_returns,
        )

        strategy_returns.append(strat_return)
        baseline_returns.append(baseline_ret)
        argmax_returns.append(argmax_ret)
        risk_adj_returns.append(risk_adj_ret)

        # Update expanding average returns for risk-adjusted (no lookahead)
        for cls in ["ATM", "OTM5", "OTM10"]:
            col = f"return_{cls}"
            if not month_br.empty and col in month_br.columns:
                vals = month_br[col].dropna()
                if not vals.empty:
                    cumulative_sums[cls] += vals.sum()
                    cumulative_counts[cls] += len(vals)
                    avg_returns[cls] = cumulative_sums[cls] / cumulative_counts[cls]

        # Track detail
        monthly_detail.append({
            "month": str(month),
            "n_tickers": len(month_data),
            "n_allocated": len(allocated),
            "strategy_return": round(strat_return, 6),
            "baseline_return": round(baseline_ret, 6),
            "argmax_return": round(argmax_ret, 6),
            "risk_adj_return": round(risk_adj_ret, 6),
            "top_picks": list(allocated["symbol"].values) if not allocated.empty else [],
        })

        # Update previous buckets for next month's turnover computation
        for _, row in month_data.iterrows():
            prev_buckets[row["symbol"]] = row["model_bucket"]

        if (i + 1) % 50 == 0:
            logger.info(f"Backtest progress: {i + 1}/{len(months)} months")

    logger.info(f"Backtest complete: {len(months)} months processed.")

    return {
        "preset": preset,
        "budget": budget,
        "n_months": len(months),
        "strategy": {
            "returns": strategy_returns,
            "metrics": compute_metrics(strategy_returns),
        },
        "baseline": {
            "name": "OTM10 all tickers, equal weight",
            "returns": baseline_returns,
            "metrics": compute_metrics(baseline_returns),
        },
        "argmax": {
            "name": "Model top pick, all tickers, equal weight",
            "returns": argmax_returns,
            "metrics": compute_metrics(argmax_returns),
        },
        "risk_adjusted": {
            "name": "P(bucket) × E[return], all tickers, equal weight",
            "returns": risk_adj_returns,
            "metrics": compute_metrics(risk_adj_returns),
        },
        "monthly_detail": monthly_detail,
    }


async def run_backtest(preset: str = "balanced", budget: float = 100_000,
                       force: bool = False) -> dict:
    """Run the backtesting pipeline for a single preset.

    Checks cache first. If no cache or force=True, runs the full
    backtest loop and caches the result.

    Args:
        preset: Strategy preset ('conservative', 'balanced', 'aggressive').
        budget: Dollar budget per month.
        force: If True, ignore cache and recompute.

    Returns:
        Dict with backtest report including metrics for strategy + baseline.
    """
    try:
        if not force:
            cached = _load_cached_report(preset)
            if cached is not None:
                cached["source"] = "cache"
                logger.info(f"Returning cached backtest for preset={preset}")
                return cached

        logger.info(f"Running backtest: preset={preset}, budget=${budget:,.0f}")
        fs = pd.read_parquet(DATA_DIR / "feature_store.parquet")
        fs["decision_date"] = pd.to_datetime(fs["decision_date"])
        fs["year_month"] = fs["decision_date"].dt.to_period("M")

        report = _run_backtest_loop(fs, budget, preset)
        report["source"] = "computed"

        _save_cached_report(report, preset)
        logger.info(f"Backtest cached for preset={preset}")

        return report

    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        return {"error": f"Backtest failed: {e}"}


async def run_backtest_all(budget: float = 100_000, year: str = "all",
                           force: bool = False) -> dict:
    """Run backtests for both models and return combined results.

    LGBM strategies: Baseline, Argmax, Risk-Adjusted, Conservative.
    LSTM strategies: Argmax, Risk-Adjusted, Conservative (same scoring engine).

    Args:
        budget: Dollar budget per month.
        year: Year string (e.g. "2020") or "all" for full history.
        force: If True, ignore cache and recompute.

    Returns:
        Dict with results for both models + baseline + year range info.
    """
    try:
        cache_key = f"dual_{year}"

        if not force:
            cached = _load_cached_report(cache_key)
            if cached is not None:
                cached["source"] = "cache"
                logger.info(f"Returning cached backtest_all for year={year}")
                return cached

        # ── LGBM feature store ──
        logger.info(f"Running backtest_all: year={year}, budget=${budget:,.0f}")
        fs = pd.read_parquet(DATA_DIR / "feature_store.parquet")
        fs["decision_date"] = pd.to_datetime(fs["decision_date"])
        fs["year_month"] = fs["decision_date"].dt.to_period("M")

        if year != "all":
            fs = fs[fs["decision_date"].dt.year == int(year)]
            if fs.empty:
                return {"error": f"No LGBM data for year {year}"}

        # LGBM: Conservative only (Balanced/Aggressive commented out)
        lgbm_report = _run_backtest_loop(fs, budget, "conservative")
        lgbm_results = {
            "conservative": {"metrics": lgbm_report["strategy"]["metrics"]},
            # "balanced": ...,     # removed — keeping only conservative
            # "aggressive": ...,   # removed — keeping only conservative
        }

        # ── LSTM feature store ──
        from src.inference.lstm_model import get_monthly_predictions
        lstm_monthly = get_monthly_predictions()
        lstm_monthly = lstm_monthly[lstm_monthly["year_month"].notna()].copy()

        if year != "all":
            lstm_monthly = lstm_monthly[
                lstm_monthly["date"].dt.year == int(year)
            ]

        # LSTM backtest: same loop, same scoring engine, different predictions
        lstm_report = None
        if not lstm_monthly.empty:
            lstm_report = _run_backtest_loop(lstm_monthly, budget, "conservative")

        # ── Combine ──
        date_range_start = str(fs["year_month"].min())
        date_range_end = str(fs["year_month"].max())

        combined = {
            "year": year,
            "budget": budget,
            "n_months": len(fs["year_month"].unique()),
            "date_range": {"start": date_range_start, "end": date_range_end},
            # LGBM strategies
            "lgbm": {
                "baseline": lgbm_report["baseline"],
                "argmax": lgbm_report["argmax"],
                "risk_adjusted": lgbm_report["risk_adjusted"],
                "conservative": lgbm_results["conservative"],
            },
            # LSTM strategies
            "lstm": {
                "argmax": lstm_report["argmax"] if lstm_report else {"metrics": {}},
                "risk_adjusted": lstm_report["risk_adjusted"] if lstm_report else {"metrics": {}},
                "conservative": {"metrics": lstm_report["strategy"]["metrics"]} if lstm_report else {"metrics": {}},
            },
            # Top-level baseline (shared — model-independent)
            "baseline": lgbm_report["baseline"],
        }

        _save_cached_report(combined, cache_key)
        combined["source"] = "computed"
        return combined

    except Exception as e:
        logger.error(f"Backtest_all failed: {e}")
        return {"error": f"Backtest_all failed: {e}"}
