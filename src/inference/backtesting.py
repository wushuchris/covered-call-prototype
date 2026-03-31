# Backtesting pipeline for the covered call strategy.
#
# Loops through historical months, applies scoring + allocation each month,
# computes realized returns, and produces a report with metrics for the
# strategy and both baselines.
#
# Cache-first: if a report exists on disk for the given preset, return it.
# Otherwise run the full backtest and cache the result.
#
# NautilusTrader integration:
#   The BacktestEngine scaffold is preserved for future use when venue,
#   instrument, and order execution simulation are wired. For now, the
#   backtest loop runs in plain Python using the scoring engine directly.
#   NautilusTrader will replace the inner loop once order-level tracking
#   is needed (fills, slippage, margin, position lifecycle).

import json
import numpy as np
import pandas as pd
from pathlib import Path

# NautilusTrader scaffold — preserved for future execution-level backtesting
# from nautilus_trader.backtest.engine import BacktestEngine
# from nautilus_trader.backtest.config import BacktestEngineConfig
# from nautilus_trader.model import TraderId
# from src.inference.strategy import CoveredCallStrategy, CoveredCallStrategyConfig

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
            moneyness = bucket.split("_")[0]
            col = f"return_{moneyness}"

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

        strategy_returns.append(strat_return)
        baseline_returns.append(baseline_ret)

        # Track detail
        monthly_detail.append({
            "month": str(month),
            "n_tickers": len(month_data),
            "n_allocated": len(allocated),
            "strategy_return": round(strat_return, 6),
            "baseline_return": round(baseline_ret, 6),
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
    """Run backtests for all 3 presets and return combined results.

    Cache-first per preset. Optionally filter to a specific year.

    Args:
        budget: Dollar budget per month.
        year: Year string (e.g. "2020") or "all" for full history.
        force: If True, ignore cache and recompute.

    Returns:
        Dict with results for all presets + baseline + year range info.
    """
    try:
        # Build cache key incorporating year
        cache_key = f"all_{year}"

        if not force:
            cached = _load_cached_report(cache_key)
            if cached is not None:
                cached["source"] = "cache"
                logger.info(f"Returning cached backtest_all for year={year}")
                return cached

        logger.info(f"Running backtest_all: year={year}, budget=${budget:,.0f}")
        fs = pd.read_parquet(DATA_DIR / "feature_store.parquet")
        fs["decision_date"] = pd.to_datetime(fs["decision_date"])
        fs["year_month"] = fs["decision_date"].dt.to_period("M")

        # Filter to year if specified
        if year != "all":
            fs = fs[fs["decision_date"].dt.year == int(year)]
            if fs.empty:
                return {"error": f"No data for year {year}"}

        presets = ["conservative", "balanced", "aggressive"]
        results = {}

        for preset in presets:
            report = _run_backtest_loop(fs, budget, preset)
            results[preset] = {
                "metrics": report["strategy"]["metrics"],
                "monthly_detail": report["monthly_detail"],
            }
            # Baseline is same for all presets (computed in each loop, take from first)
            if "baseline" not in results:
                results["baseline"] = report["baseline"]

        combined = {
            "year": year,
            "budget": budget,
            "n_months": len(fs["year_month"].unique()),
            "date_range": {
                "start": str(fs["year_month"].min()),
                "end": str(fs["year_month"].max()),
            },
            "presets": results,
            "baseline": results["baseline"],
        }

        _save_cached_report(combined, cache_key)
        combined["source"] = "computed"
        return combined

    except Exception as e:
        logger.error(f"Backtest_all failed: {e}")
        return {"error": f"Backtest_all failed: {e}"}


# ── NautilusTrader scaffold ─────────────────────────────────────────────────
# Preserved for future execution-level backtesting.
# When venue/instruments are wired, the _run_backtest_loop above will be
# replaced by:
#
#   1. BacktestEngine(config) with venue + instruments
#   2. CoveredCallStrategy that calls score_month() in on_bar()
#   3. engine.run() → extract fills, positions, account reports
#   4. NautilusTrader handles order execution, slippage, margin natively
#
# def _build_engine() -> BacktestEngine:
#     config = BacktestEngineConfig(trader_id=TraderId("BACKTESTER-001"))
#     engine = BacktestEngine(config=config)
#     # engine.add_venue(...)
#     # engine.add_instrument(...)
#     # engine.add_data(...)
#     # engine.add_strategy(CoveredCallStrategy(config))
#     return engine
