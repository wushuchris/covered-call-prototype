# Composable scoring engine for covered call strategy.
#
# Sits between model.py (fixed predictions) and strategy.py (execution).
# Ranks the 10-ticker universe each month using three score components:
#   1. Model confidence (LGBM predict_proba)
#   2. TC-adjusted score (bid-ask spread + turnover penalty)
#   3. Delta-hedged return score (vol premium after removing directional exposure)
#
# Three presets control the weights: Conservative / Balanced / Aggressive.
# Capital allocation selects top N tickers by composite score.
#
# References:
#   - TC model: Tan et al. 2024 (arXiv 2407.21791)
#   - Delta-hedged return: Bali et al. 2023 (SSRN 3895984), monthly approximation

import numpy as np
import pandas as pd
from pathlib import Path

from src.utils import create_logger

logger = create_logger("scoring")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = _PROJECT_ROOT / "data" / "processed"

# ── Presets ───────────────────────────────────────────────────────────────────
# (w_confidence, w_tc, w_delta_hedge)

PRESETS = {
    "conservative": (0.3, 0.5, 0.2),
    "balanced":     (0.33, 0.33, 0.34),
    "aggressive":   (0.6, 0.1, 0.3),
}

# Conservative: spread capital wide, avoid costly positions
# Balanced: equal weight to all signals
# Aggressive: chase model confidence, tolerate higher costs

# ── Allocation rules per preset ──────────────────────────────────────────────
# (max_positions, sizing_method)
#   equal: divide budget equally across selected positions
#   proportional: allocate proportional to composite score

ALLOCATION = {
    "conservative": {"max_positions": 7, "sizing": "equal"},
    "balanced":     {"max_positions": 5, "sizing": "equal"},
    "aggressive":   {"max_positions": 3, "sizing": "proportional"},
}

# Risk-free rate approximation (annualized, for delta-hedge financing cost)
RISK_FREE_RATE = 0.04

# ── Options data (lazy-loaded singleton) ─────────────────────────────────────
_options_cache = None


def _load_options():
    """Load and prepare options data for spread and delta-hedge computations.

    Filters to calls only, computes DTE, year_month, and spread.
    Cached as a module-level singleton.

    Returns:
        DataFrame of call options with spread and bucket assignment columns.
    """
    global _options_cache
    if _options_cache is not None:
        return _options_cache

    try:
        opts = pd.read_parquet(DATA_DIR / "options_clean.parquet")
        calls = opts[opts["call_put"] == "CALL"].copy()
        calls["trade_date"] = pd.to_datetime(calls["trade_date"])
        calls["expiration"] = pd.to_datetime(calls["expiration"])
        calls["dte"] = (calls["expiration"] - calls["trade_date"]).dt.days
        calls["year_month"] = calls["trade_date"].dt.to_period("M")
        calls["spread"] = calls["ask"] - calls["bid"]
        calls["spread_pct"] = calls["spread"] / calls["mark"]

        # Assign moneyness bucket by delta
        calls["moneyness"] = pd.cut(
            calls["delta"],
            bins=[0.15, 0.30, 0.45, 0.60],
            labels=["OTM10", "OTM5", "ATM"],
        )

        # Assign maturity bucket by DTE
        calls["maturity"] = np.where(calls["dte"] <= 45, "SHORT", "LONG")

        calls = calls.dropna(subset=["moneyness"])
        _options_cache = calls
        logger.info(f"Options loaded for scoring: {len(calls)} call contracts.")
        return calls

    except Exception as e:
        logger.error(f"Failed to load options for scoring: {e}")
        raise


# ── Score component 1: Confidence ─────────────────────────────────────────────

def _confidence_scores(month_data: pd.DataFrame) -> pd.Series:
    """Extract model confidence as a score. Already in [0, 1].

    Args:
        month_data: Feature store rows for a single month.

    Returns:
        Series indexed like month_data with confidence scores.
    """
    return month_data["model_confidence"].values


# ── Score component 2: TC-adjusted ───────────────────────────────────────────

def _tc_scores(month_data: pd.DataFrame, prev_buckets: dict) -> np.ndarray:
    """Compute transaction cost score per ticker for a given month.

    Lower spread + no bucket change = higher score.

    Args:
        month_data: Feature store rows for a single month.
        prev_buckets: Dict of {ticker: bucket_string} from previous month.

    Returns:
        Array of TC scores in [0, 1], same length as month_data.
    """
    try:
        calls = _load_options()
        period = month_data["year_month"].iloc[0]
        scores = []

        for _, row in month_data.iterrows():
            ticker = row["symbol"]
            moneyness = row["model_moneyness"]
            maturity = row["model_maturity"]
            bucket = row["model_bucket"]

            # Find matching contracts for this ticker, month, moneyness
            mask = (
                (calls["symbol"] == ticker)
                & (calls["year_month"] == period)
                & (calls["moneyness"] == moneyness)
            )
            if maturity == "SHORT":
                mask = mask & (calls["dte"] <= 45)
            else:
                mask = mask & (calls["dte"] > 45)

            matching = calls[mask]

            if matching.empty:
                # No matching contracts — penalize with high cost
                spread_cost = 1.0
            else:
                # Average spread as fraction of option price
                spread_cost = matching["spread_pct"].clip(0, 1).mean()

            # Turnover penalty: double the cost if bucket changed
            prev_bucket = prev_buckets.get(ticker)
            if prev_bucket is not None and prev_bucket != bucket:
                spread_cost *= 2.0

            scores.append(spread_cost)

        # Invert: lower cost = higher score, normalize to [0, 1]
        costs = np.array(scores)
        max_cost = costs.max() if costs.max() > 0 else 1.0
        tc_scores = 1.0 - (costs / max_cost)
        return tc_scores

    except Exception as e:
        logger.error(f"TC score computation failed: {e}")
        return np.full(len(month_data), 0.5)


# ── Score component 3: Delta-hedged return ───────────────────────────────────

def _delta_hedge_scores(month_data: pd.DataFrame, daily_prices: pd.DataFrame) -> np.ndarray:
    """Compute delta-hedged return score per ticker for a given month.

    Monthly approximation of Bali et al. 2023:
        DH_gain = option_pnl - delta * stock_move - financing_cost

    Args:
        month_data: Feature store rows for a single month.
        daily_prices: Daily stock prices (needs symbol, date, adjusted_close).

    Returns:
        Array of delta-hedge scores in [0, 1], same length as month_data.
    """
    try:
        calls = _load_options()
        period = month_data["year_month"].iloc[0]

        # Next month's period for exit prices
        next_period = period + 1

        dh_returns = []

        for _, row in month_data.iterrows():
            ticker = row["symbol"]
            moneyness = row["model_moneyness"]
            maturity = row["model_maturity"]

            # Entry: matching contracts this month
            entry_mask = (
                (calls["symbol"] == ticker)
                & (calls["year_month"] == period)
                & (calls["moneyness"] == moneyness)
            )
            if maturity == "SHORT":
                entry_mask = entry_mask & (calls["dte"] <= 45)
            else:
                entry_mask = entry_mask & (calls["dte"] > 45)

            entry_contracts = calls[entry_mask]

            # Exit: same ticker next month (approximate — find similar contracts)
            exit_mask = (
                (calls["symbol"] == ticker)
                & (calls["year_month"] == next_period)
                & (calls["moneyness"] == moneyness)
            )
            exit_contracts = calls[exit_mask]

            if entry_contracts.empty or exit_contracts.empty:
                dh_returns.append(0.0)
                continue

            # Use average mark prices as proxy
            entry_mark = entry_contracts["mark"].mean()
            exit_mark = exit_contracts["mark"].mean()
            entry_delta = entry_contracts["delta"].mean()

            # Stock prices at entry and exit months
            ticker_daily = daily_prices[daily_prices["symbol"] == ticker]
            entry_prices = ticker_daily[
                ticker_daily["date"].dt.to_period("M") == period
            ]
            exit_prices = ticker_daily[
                ticker_daily["date"].dt.to_period("M") == next_period
            ]

            if entry_prices.empty or exit_prices.empty:
                dh_returns.append(0.0)
                continue

            stock_entry = entry_prices["adjusted_close"].iloc[-1]
            stock_exit = exit_prices["adjusted_close"].iloc[-1]

            # Delta-hedged gain (monthly approximation of Bali et al.)
            option_pnl = exit_mark - entry_mark
            hedge_pnl = entry_delta * (stock_exit - stock_entry)
            financing = (RISK_FREE_RATE / 12) * (entry_mark - entry_delta * stock_entry)
            dh_gain = option_pnl - hedge_pnl - financing

            # Normalize by capital deployed
            capital = abs(entry_delta * stock_entry - entry_mark)
            dh_return = dh_gain / capital if capital > 0 else 0.0
            dh_returns.append(dh_return)

        # Normalize to [0, 1]
        arr = np.array(dh_returns)
        if arr.max() == arr.min():
            return np.full(len(arr), 0.5)
        return (arr - arr.min()) / (arr.max() - arr.min())

    except Exception as e:
        logger.error(f"Delta-hedge score computation failed: {e}")
        return np.full(len(month_data), 0.5)


# ── Daily prices (lazy-loaded) ───────────────────────────────────────────────
_daily_cache = None


def _load_daily():
    """Load daily price data for delta-hedge computation.

    Returns:
        DataFrame with symbol, date, adjusted_close.
    """
    global _daily_cache
    if _daily_cache is not None:
        return _daily_cache

    try:
        df = pd.read_parquet(DATA_DIR / "daily_clean.parquet")
        df["date"] = pd.to_datetime(df["date"])
        _daily_cache = df
        logger.info(f"Daily prices loaded for scoring: {len(df)} rows.")
        return df

    except Exception as e:
        logger.error(f"Failed to load daily prices: {e}")
        raise


# ── Composite scoring ────────────────────────────────────────────────────────

def score_month(month_data: pd.DataFrame, prev_buckets: dict,
                preset: str = "balanced") -> pd.DataFrame:
    """Score all tickers for a single month using the selected preset.

    Args:
        month_data: Feature store rows for one month (all tickers).
        prev_buckets: Dict {ticker: bucket} from previous month for turnover.
        preset: One of 'conservative', 'balanced', 'aggressive'.

    Returns:
        DataFrame with columns: symbol, model_bucket, confidence_score,
        tc_score, dh_score, composite_score — sorted by composite descending.
    """
    try:
        w_conf, w_tc, w_dh = PRESETS.get(preset, PRESETS["balanced"])
        daily = _load_daily()

        conf = _confidence_scores(month_data)
        tc = _tc_scores(month_data, prev_buckets)
        dh = _delta_hedge_scores(month_data, daily)

        composite = w_conf * conf + w_tc * tc + w_dh * dh

        result = pd.DataFrame({
            "symbol": month_data["symbol"].values,
            "model_bucket": month_data["model_bucket"].values,
            "model_confidence": conf,
            "best_bucket": month_data["best_bucket"].values,
            "best_return": month_data["best_return"].values,
            "confidence_score": conf,
            "tc_score": tc,
            "dh_score": dh,
            "composite_score": composite,
        })

        return result.sort_values("composite_score", ascending=False).reset_index(drop=True)

    except Exception as e:
        logger.error(f"score_month failed: {e}")
        raise


# ── Capital allocation ───────────────────────────────────────────────────────

def allocate(scored: pd.DataFrame, budget: float,
             preset: str = "balanced") -> pd.DataFrame:
    """Allocate capital to top-ranked tickers based on preset rules.

    Args:
        scored: Output of score_month(), sorted by composite_score desc.
        budget: Total dollar budget available.
        preset: Preset name for allocation rules.

    Returns:
        DataFrame with allocated tickers, including 'allocation' column (dollars).
    """
    try:
        rules = ALLOCATION.get(preset, ALLOCATION["balanced"])
        max_pos = rules["max_positions"]
        sizing = rules["sizing"]

        # Take top N
        selected = scored.head(max_pos).copy()

        if sizing == "proportional":
            total_score = selected["composite_score"].sum()
            if total_score > 0:
                selected["allocation"] = budget * (selected["composite_score"] / total_score)
            else:
                selected["allocation"] = budget / len(selected)
        else:
            # Equal weight
            selected["allocation"] = budget / len(selected)

        selected["allocation"] = selected["allocation"].round(2)
        return selected.reset_index(drop=True)

    except Exception as e:
        logger.error(f"allocate failed: {e}")
        raise


# ── Baselines ────────────────────────────────────────────────────────────────

def baseline_no_model(month_data: pd.DataFrame, budget: float) -> pd.DataFrame:
    """Baseline 1: OTM10_SHORT on all tickers, equal weight.

    No model, no scoring. Just sell 10% OTM short-dated calls on everything.

    Args:
        month_data: Feature store rows for one month.
        budget: Total dollar budget.

    Returns:
        DataFrame with all tickers, fixed bucket, equal allocation.
    """
    result = pd.DataFrame({
        "symbol": month_data["symbol"].values,
        "model_bucket": "OTM10_SHORT",
        "composite_score": 0.0,
        "allocation": budget / len(month_data),
    })
    return result


def baseline_no_scoring(month_data: pd.DataFrame, budget: float) -> pd.DataFrame:
    """Baseline 2: Model buckets on all tickers, equal weight.

    Uses the LGBM model predictions but no scoring or selection.

    Args:
        month_data: Feature store rows for one month.
        budget: Total dollar budget.

    Returns:
        DataFrame with all tickers, model buckets, equal allocation.
    """
    result = pd.DataFrame({
        "symbol": month_data["symbol"].values,
        "model_bucket": month_data["model_bucket"].values,
        "composite_score": 0.0,
        "allocation": budget / len(month_data),
    })
    return result


# ── Metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(monthly_returns: list[float]) -> dict:
    """Compute the 5 primary strategy metrics from a series of monthly returns.

    Metrics (Tan et al. 2024):
        1. Cumulative return
        2. Sharpe ratio (annualized)
        3. Max drawdown
        4. Hit rate (% positive months)
        5. Ave. P / Ave. L

    Args:
        monthly_returns: List of monthly return values.

    Returns:
        Dict with metric names and values.
    """
    try:
        r = np.array(monthly_returns)

        if len(r) == 0:
            return _empty_metrics()

        # Cumulative return (as total %)
        cumulative = float(np.prod(1 + r) - 1)

        # Annualized return (CAGR)
        n_years = len(r) / 12
        annualized = float((1 + cumulative) ** (1 / n_years) - 1) if n_years > 0 else 0.0

        # Annualized Sharpe (monthly returns → annualized)
        mean_r = r.mean()
        std_r = r.std()
        sharpe = float((mean_r / std_r) * np.sqrt(12)) if std_r > 0 else 0.0

        # Max drawdown
        cumulative_curve = np.cumprod(1 + r)
        running_max = np.maximum.accumulate(cumulative_curve)
        drawdowns = (cumulative_curve - running_max) / running_max
        max_dd = float(drawdowns.min())

        # Hit rate
        hit_rate = float((r > 0).mean())

        # Ave P / Ave L
        gains = r[r > 0]
        losses = r[r < 0]
        avg_gain = float(gains.mean()) if len(gains) > 0 else 0.0
        avg_loss = float(abs(losses.mean())) if len(losses) > 0 else 1.0
        avg_p_l = avg_gain / avg_loss if avg_loss > 0 else 0.0

        return {
            "annualized_return": round(annualized, 4),
            "sharpe_ratio": round(sharpe, 4),
            "max_drawdown": round(max_dd, 4),
            "hit_rate": round(hit_rate, 4),
            "avg_p_l": round(avg_p_l, 4),
        }

    except Exception as e:
        logger.error(f"compute_metrics failed: {e}")
        return _empty_metrics()


def _empty_metrics() -> dict:
    """Return zeroed metrics dict."""
    return {
        "annualized_return": 0.0,
        "sharpe_ratio": 0.0,
        "max_drawdown": 0.0,
        "hit_rate": 0.0,
        "avg_p_l": 0.0,
    }
