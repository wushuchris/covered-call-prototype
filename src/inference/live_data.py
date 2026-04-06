"""
Live data pipeline — fetches real-time market data for today's inference.

Called when the requested date is beyond the precomputed feature stores.
Uses yfinance (prices, options, fundamentals), scipy (Black-Scholes delta),
and fredapi (macro features for LSTM).

Returns:
    - lgbm_features: dict of 34 features for LGBM prediction
    - lstm_window: (50, 35) numpy array for LSTM-CNN prediction
    - contracts: DataFrame for the scoring engine (TC + delta-hedge)
"""

import os
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
from pathlib import Path

from src.utils import create_logger

logger = create_logger("live_data")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = _PROJECT_ROOT / "data" / "processed"

UNIVERSE = ["AAPL", "AMZN", "AVGO", "GOOG", "GOOGL", "META", "MSFT", "NVDA", "TSLA", "WMT"]

# LGBM expects these 34 features
LGBM_FEATURES = [
    "vol_21d", "vol_63d", "mom_5d", "mom_21d", "mom_63d",
    "price_to_sma21", "price_to_sma200", "sma21_above_sma50",
    "sma50_above_sma200", "drawdown_63d", "drawdown_252d",
    "volume_ratio", "high_vol_regime", "gross_margin", "net_margin",
    "revenue_growth_yoy", "earnings_growth_yoy", "debt_to_equity",
    "cash_ratio", "roe", "roa", "pe_ratio", "ps_ratio", "ev_ebitda",
    "fcf_yield", "iv_mean", "iv_median", "iv_skew", "iv_short_mean",
    "iv_short_std", "iv_long_mean", "iv_term_structure", "iv_rank",
    "iv_change",
]

# LSTM expects these 35 features
LSTM_FEATURES = [
    "CPIAUCSL", "ma_200", "longTermDebt", "dividendPayout", "gross_margin",
    "ma_50", "totalLiabilities", "totalCurrentAssets", "ma_20", "rolling_max",
    "book_value_per_share_proxy", "open", "Beta", "UNRATE", "FEDFUNDS",
    "BookValue", "50DayMovingAverage", "low", "DGS10", "high", "close",
    "adj_close", "52WeekHigh", "commonStockSharesOutstanding",
    "SharesOutstanding", "operatingExpenses", "EPS", "NFCI", "totalAssets",
    "totalCurrentLiabilities", "52WeekLow", "200DayMovingAverage",
    "yield_curve", "totalShareholderEquity", "debt_to_equity",
]

SEQ_LEN = 50
RISK_FREE_RATE = 0.04


# ── Black-Scholes delta ─────────────────────────────────────────────────

def _bs_delta(S, K, T, r, sigma):
    """Compute Black-Scholes call delta.

    Args:
        S: Spot price. K: Strike. T: Time to expiry (years).
        r: Risk-free rate. sigma: Implied volatility.

    Returns:
        Call delta in [0, 1].
    """
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.5
    d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    return float(norm.cdf(d1))


# ── FRED macro data ──────────────────────────────────────────────────────

_fred_cache = None


def _fetch_fred() -> dict:
    """Fetch latest FRED macro values. Cached for the session.

    Returns:
        Dict with FEDFUNDS, DGS10, DGS2, yield_curve, CPIAUCSL, UNRATE, NFCI.
    """
    global _fred_cache
    if _fred_cache is not None:
        return _fred_cache

    try:
        from fredapi import Fred

        api_key = os.getenv("FRED_API_KEY", "")
        if not api_key:
            env_path = _PROJECT_ROOT / "src" / ".env"
            if env_path.exists():
                for line in env_path.read_text().splitlines():
                    if line.startswith("FRED_API_KEY"):
                        api_key = line.split("=", 1)[1].strip().strip("'\"")

        if not api_key:
            logger.warning("No FRED_API_KEY — using fallback macro values")
            _fred_cache = _fred_fallback()
            return _fred_cache

        fred = Fred(api_key=api_key)
        result = {}
        for series in ["FEDFUNDS", "DGS10", "DGS2", "CPIAUCSL", "UNRATE", "NFCI"]:
            try:
                s = fred.get_series(series, observation_start="2025-01-01")
                result[series] = float(s.dropna().iloc[-1])
            except Exception:
                result[series] = 0.0

        result["yield_curve"] = result.get("DGS10", 0) - result.get("DGS2", 0)
        _fred_cache = result
        logger.info(f"FRED data fetched: {result}")
        return result

    except Exception as e:
        logger.warning(f"FRED fetch failed: {e} — using fallback")
        _fred_cache = _fred_fallback()
        return _fred_cache


def _fred_fallback() -> dict:
    """Static fallback macro values (approximate recent values)."""
    return {
        "FEDFUNDS": 4.33, "DGS10": 4.25, "DGS2": 3.95,
        "yield_curve": 0.30, "CPIAUCSL": 313.0, "UNRATE": 4.2, "NFCI": -0.5,
    }


# ── yfinance data fetching ───────────────────────────────────────────────

_batch_prices_cache = {}


def fetch_batch_prices(tickers: list, days: int = 400) -> dict:
    """Bulk-download daily prices for all tickers in one yfinance call.

    Caches for the session (cleared on server restart).

    Args:
        tickers: List of symbols.
        days: Calendar days of history.

    Returns:
        Dict mapping ticker → DataFrame (date, open, high, low, close, volume).
    """
    global _batch_prices_cache
    cache_key = tuple(sorted(tickers))
    if cache_key in _batch_prices_cache:
        return _batch_prices_cache[cache_key]

    try:
        raw = yf.download(tickers, period=f"{days}d", auto_adjust=True,
                          group_by="ticker", threads=True)
        result = {}
        for t in tickers:
            try:
                df = raw[t].dropna(how="all").reset_index()
                df.columns = [c.lower() for c in df.columns]
                result[t] = df
            except Exception:
                result[t] = pd.DataFrame()
        _batch_prices_cache[cache_key] = result
        logger.info(f"Batch prices fetched: {len(result)} tickers")
        return result
    except Exception as e:
        logger.error(f"Batch price fetch failed: {e}")
        return {t: pd.DataFrame() for t in tickers}


def _get_fundamentals(ticker: str) -> dict:
    """Fetch fundamentals from balance_sheet, financials, and info.

    Uses the structured financial statements (not just .info) to get
    values matching the Alpha Vantage schema the models were trained on.

    Args:
        ticker: Stock symbol.

    Returns:
        Dict with fundamental values.
    """
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}
        bs = t.balance_sheet
        inc = t.financials

        # Latest quarter from balance sheet
        bs_latest = bs.iloc[:, 0] if not bs.empty else pd.Series(dtype=float)
        inc_latest = inc.iloc[:, 0] if not inc.empty else pd.Series(dtype=float)

        def _get(series, key, default=0.0):
            v = series.get(key, default)
            return float(v) if v is not None and not pd.isna(v) else default

        return {
            "totalAssets": _get(bs_latest, "Total Assets"),
            "totalCurrentAssets": _get(bs_latest, "Current Assets"),
            "totalLiabilities": _get(bs_latest, "Total Liabilities Net Minority Interest"),
            "totalCurrentLiabilities": _get(bs_latest, "Current Liabilities"),
            "longTermDebt": _get(bs_latest, "Long Term Debt"),
            "totalShareholderEquity": _get(bs_latest, "Stockholders Equity"),
            "commonStockSharesOutstanding": float(info.get("sharesOutstanding", 0) or 0),
            "SharesOutstanding": float(info.get("sharesOutstanding", 0) or 0),
            "operatingExpenses": _get(inc_latest, "Operating Expense"),
            "dividendPayout": float(info.get("lastDividendValue", 0) or 0),
            "EPS": float(info.get("trailingEps", 0) or 0),
            "BookValue": float(info.get("bookValue", 0) or 0),
            "Beta": float(info.get("beta", 1) or 1),
            # Derived
            "grossMargins": float(info.get("grossMargins", 0) or 0),
            "profitMargins": float(info.get("profitMargins", 0) or 0),
            "revenueGrowth": float(info.get("revenueGrowth", 0) or 0),
            "earningsGrowth": float(info.get("earningsGrowth", 0) or 0),
            "debtToEquity": float(info.get("debtToEquity", 0) or 0),
            "trailingPE": float(info.get("trailingPE", 0) or 0),
            "priceToSalesTrailing12Months": float(info.get("priceToSalesTrailing12Months", 0) or 0),
            "enterpriseToEbitda": float(info.get("enterpriseToEbitda", 0) or 0),
            "freeCashflow": float(info.get("freeCashflow", 0) or 0),
            "marketCap": float(info.get("marketCap", 1) or 1),
            "netIncomeToCommon": float(info.get("netIncomeToCommon", 0) or 0),
            # yfinance 52-week stats from info (updated daily)
            "52WeekHigh": float(info.get("fiftyTwoWeekHigh", 0) or 0),
            "52WeekLow": float(info.get("fiftyTwoWeekLow", 0) or 0),
            "50DayMovingAverage": float(info.get("fiftyDayAverage", 0) or 0),
            "200DayMovingAverage": float(info.get("twoHundredDayAverage", 0) or 0),
        }

    except Exception as e:
        logger.warning(f"Fundamentals fetch failed for {ticker}: {e}")
        return {}


def _fetch_ticker_data(ticker: str, days: int = 400) -> dict:
    """Fetch prices, fundamentals, and options for one ticker.

    Uses batch-cached prices if available, otherwise fetches individually.

    Args:
        ticker: Stock symbol.
        days: Number of calendar days of history to fetch.

    Returns:
        Dict with 'prices' (DataFrame), 'fundamentals' (dict), 'options' (DataFrame).
    """
    try:
        # Check batch cache first
        cache_key = tuple(sorted(UNIVERSE))
        if cache_key in _batch_prices_cache and ticker in _batch_prices_cache[cache_key]:
            prices = _batch_prices_cache[cache_key][ticker]
        else:
            t = yf.Ticker(ticker)
            prices = t.history(period=f"{days}d", auto_adjust=True).reset_index()
            prices.columns = [c.lower() for c in prices.columns]

        fundamentals = _get_fundamentals(ticker)

        # Options - nearest expiry
        options_df = pd.DataFrame()
        try:
            t = yf.Ticker(ticker)
            expiries = t.options
            if expiries:
                chain = t.option_chain(expiries[0])
                calls = chain.calls.copy()
                calls["symbol"] = ticker
                calls["expiration"] = pd.Timestamp(expiries[0])
                calls["trade_date"] = pd.Timestamp.now().normalize()
                options_df = calls
        except Exception as e:
            logger.warning(f"Options fetch failed for {ticker}: {e}")

        return {"prices": prices, "fundamentals": fundamentals, "options": options_df}

    except Exception as e:
        logger.error(f"yfinance fetch failed for {ticker}: {e}")
        return {"prices": pd.DataFrame(), "fundamentals": {}, "options": pd.DataFrame()}


# ── LGBM feature computation ────────────────────────────────────────────

def _compute_lgbm_features(prices: pd.DataFrame, fundamentals: dict,
                           options: pd.DataFrame) -> dict:
    """Compute 34 LGBM features from live data.

    Args:
        prices: Daily OHLCV DataFrame (needs ~250 rows for rolling windows).
        fundamentals: Dict from _get_fundamentals().
        options: Call options DataFrame with impliedVolatility.

    Returns:
        Dict with feature names as keys, scalar values.
    """
    if prices.empty or len(prices) < 63:
        return {}

    close = prices["close"].values
    volume = prices["volume"].values
    returns = pd.Series(close).pct_change().values

    feats = {}
    feats["vol_21d"] = float(np.std(returns[-21:]) * np.sqrt(252))
    feats["vol_63d"] = float(np.std(returns[-63:]) * np.sqrt(252))
    feats["mom_5d"] = float(close[-1] / close[-5] - 1) if len(close) >= 5 else 0.0
    feats["mom_21d"] = float(close[-1] / close[-21] - 1) if len(close) >= 21 else 0.0
    feats["mom_63d"] = float(close[-1] / close[-63] - 1) if len(close) >= 63 else 0.0

    sma21 = float(np.mean(close[-21:]))
    sma50 = float(np.mean(close[-50:])) if len(close) >= 50 else sma21
    sma200 = float(np.mean(close[-200:])) if len(close) >= 200 else sma50

    feats["price_to_sma21"] = close[-1] / sma21 if sma21 > 0 else 1.0
    feats["price_to_sma200"] = close[-1] / sma200 if sma200 > 0 else 1.0
    feats["sma21_above_sma50"] = float(sma21 > sma50)
    feats["sma50_above_sma200"] = float(sma50 > sma200)

    peak_63 = float(np.max(close[-63:])) if len(close) >= 63 else close[-1]
    peak_252 = float(np.max(close[-252:])) if len(close) >= 252 else peak_63
    feats["drawdown_63d"] = float(close[-1] / peak_63 - 1)
    feats["drawdown_252d"] = float(close[-1] / peak_252 - 1)

    vol_avg = float(np.mean(volume[-20:])) if len(volume) >= 20 else 1.0
    feats["volume_ratio"] = float(volume[-1] / vol_avg) if vol_avg > 0 else 1.0
    feats["high_vol_regime"] = float(feats["vol_21d"] > 0.25)

    # Fundamentals from balance_sheet + financials + info
    f = fundamentals
    feats["gross_margin"] = f.get("grossMargins", 0)
    feats["net_margin"] = f.get("profitMargins", 0)
    feats["revenue_growth_yoy"] = f.get("revenueGrowth", 0)
    feats["earnings_growth_yoy"] = f.get("earningsGrowth", 0)
    feats["debt_to_equity"] = f.get("debtToEquity", 0) / 100.0
    feats["cash_ratio"] = 0.0
    total_assets = f.get("totalAssets", 1) or 1
    net_income = f.get("netIncomeToCommon", 0)
    equity = f.get("totalShareholderEquity", 1) or 1
    feats["roe"] = net_income / equity if equity != 0 else 0.0
    feats["roa"] = net_income / total_assets if total_assets != 0 else 0.0
    feats["pe_ratio"] = f.get("trailingPE", 0)
    feats["ps_ratio"] = f.get("priceToSalesTrailing12Months", 0)
    feats["ev_ebitda"] = f.get("enterpriseToEbitda", 0)
    fcf = f.get("freeCashflow", 0)
    mkt_cap = f.get("marketCap", 1) or 1
    feats["fcf_yield"] = fcf / mkt_cap if mkt_cap > 0 else 0.0

    # IV features from options
    if not options.empty and "impliedVolatility" in options.columns:
        iv = options["impliedVolatility"].dropna()
        feats["iv_mean"] = float(iv.mean()) if len(iv) > 0 else 0.0
        feats["iv_median"] = float(iv.median()) if len(iv) > 0 else 0.0
        feats["iv_skew"] = float(iv.skew()) if len(iv) > 2 else 0.0

        dte = (options["expiration"] - options["trade_date"]).dt.days
        short = iv[dte <= 45]
        long = iv[dte > 45]
        feats["iv_short_mean"] = float(short.mean()) if len(short) > 0 else feats["iv_mean"]
        feats["iv_short_std"] = float(short.std()) if len(short) > 1 else 0.0
        feats["iv_long_mean"] = float(long.mean()) if len(long) > 0 else feats["iv_mean"]
        feats["iv_term_structure"] = feats["iv_long_mean"] - feats["iv_short_mean"]
        feats["iv_rank"] = 0.5  # can't compute percentile without history
        feats["iv_change"] = 0.0  # can't compute MoM without prior month
    else:
        for k in ["iv_mean", "iv_median", "iv_skew", "iv_short_mean",
                   "iv_short_std", "iv_long_mean", "iv_term_structure",
                   "iv_rank", "iv_change"]:
            feats[k] = 0.0

    return feats


# ── LSTM feature computation ────────────────────────────────────────────

def _compute_lstm_daily_row(prices_row: pd.Series, prices_df: pd.DataFrame,
                            idx: int, fundamentals: dict, fred: dict) -> dict:
    """Compute one daily row of LSTM features.

    Args:
        prices_row: Single row from prices DataFrame.
        prices_df: Full prices DataFrame (for rolling computations).
        idx: Index of this row in prices_df.
        fundamentals: Dict from _get_fundamentals().
        fred: FRED macro values dict.

    Returns:
        Dict with 35 LSTM feature values.
    """
    close = prices_df["close"].values[:idx + 1]
    f = fundamentals

    feats = {}
    feats["open"] = float(prices_row.get("open", 0))
    feats["high"] = float(prices_row.get("high", 0))
    feats["low"] = float(prices_row.get("low", 0))
    feats["close"] = float(prices_row.get("close", 0))
    feats["adj_close"] = feats["close"]

    feats["ma_20"] = float(np.mean(close[-20:])) if len(close) >= 20 else feats["close"]
    feats["ma_50"] = float(np.mean(close[-50:])) if len(close) >= 50 else feats["close"]
    feats["ma_200"] = float(np.mean(close[-200:])) if len(close) >= 200 else feats["close"]
    feats["50DayMovingAverage"] = f.get("50DayMovingAverage", feats["ma_50"])
    feats["200DayMovingAverage"] = f.get("200DayMovingAverage", feats["ma_200"])
    feats["rolling_max"] = float(np.max(close[-252:])) if len(close) >= 252 else float(np.max(close))
    feats["52WeekHigh"] = f.get("52WeekHigh", feats["rolling_max"])
    feats["52WeekLow"] = f.get("52WeekLow", float(np.min(close[-252:])) if len(close) >= 252 else float(np.min(close)))

    # Fundamentals from balance_sheet + financials
    feats["totalAssets"] = f.get("totalAssets", 0)
    feats["totalCurrentAssets"] = f.get("totalCurrentAssets", 0)
    feats["totalLiabilities"] = f.get("totalLiabilities", 0)
    feats["totalCurrentLiabilities"] = f.get("totalCurrentLiabilities", 0)
    feats["longTermDebt"] = f.get("longTermDebt", 0)
    feats["totalShareholderEquity"] = f.get("totalShareholderEquity", 0)
    feats["commonStockSharesOutstanding"] = f.get("commonStockSharesOutstanding", 0)
    feats["SharesOutstanding"] = f.get("SharesOutstanding", 0)
    feats["operatingExpenses"] = f.get("operatingExpenses", 0)
    feats["dividendPayout"] = f.get("dividendPayout", 0)
    feats["EPS"] = f.get("EPS", 0)
    feats["BookValue"] = f.get("BookValue", 0)
    feats["Beta"] = f.get("Beta", 1)
    feats["gross_margin"] = f.get("grossMargins", 0)
    feats["book_value_per_share_proxy"] = feats["BookValue"] if feats["BookValue"] > 0 else 1.0
    equity = feats["totalShareholderEquity"] if feats["totalShareholderEquity"] > 0 else 1.0
    feats["debt_to_equity"] = feats["totalLiabilities"] / equity

    # FRED macro
    feats["FEDFUNDS"] = fred.get("FEDFUNDS", 0)
    feats["DGS10"] = fred.get("DGS10", 0)
    feats["CPIAUCSL"] = fred.get("CPIAUCSL", 0)
    feats["UNRATE"] = fred.get("UNRATE", 0)
    feats["NFCI"] = fred.get("NFCI", 0)
    feats["yield_curve"] = fred.get("yield_curve", 0)

    return feats


# ── Contracts for scoring engine ─────────────────────────────────────────

def _build_contracts(ticker: str, options: pd.DataFrame,
                     spot: float) -> pd.DataFrame:
    """Transform yfinance options into scoring-engine-compatible DataFrame.

    Computes Black-Scholes delta and assigns moneyness/maturity buckets.

    Args:
        ticker: Stock symbol.
        options: yfinance call options DataFrame.
        spot: Current stock price.

    Returns:
        DataFrame matching the schema expected by scoring.py.
    """
    if options.empty:
        return pd.DataFrame()

    df = options.copy()
    df["dte"] = (df["expiration"] - df["trade_date"]).dt.days
    df = df[df["dte"] > 0].copy()

    # Black-Scholes delta
    df["delta"] = df.apply(
        lambda r: _bs_delta(
            spot, r["strike"], r["dte"] / 365.0,
            RISK_FREE_RATE, r.get("impliedVolatility", 0.3)
        ), axis=1,
    )

    df["mark"] = (df["bid"] + df["ask"]) / 2
    df["mark"] = df["mark"].clip(lower=0.01)
    df["spread"] = df["ask"] - df["bid"]
    df["spread_pct"] = df["spread"] / df["mark"]
    df["implied_vol"] = df["impliedVolatility"]
    df["call_put"] = "CALL"
    df["year_month"] = df["trade_date"].dt.to_period("M")

    # Moneyness buckets by delta
    df["moneyness"] = pd.cut(
        df["delta"], bins=[0.15, 0.30, 0.45, 0.60],
        labels=["OTM10", "OTM5", "ATM"],
    )
    df["maturity"] = np.where(df["dte"] <= 45, "SHORT", "LONG")
    df = df.dropna(subset=["moneyness"])

    return df


# ── Main entry points ────────────────────────────────────────────────────

def prefetch_batch_prices(tickers: list = None):
    """Pre-download prices for all tickers in one call.

    Call this before looping fetch_live() per ticker so that
    individual fetches hit the cache instead of making 10 calls.
    """
    tickers = tickers or UNIVERSE
    fetch_batch_prices(tickers, days=400)


def fetch_live(ticker: str) -> dict:
    """Fetch and compute everything needed for live inference on one ticker.

    Args:
        ticker: Stock symbol.

    Returns:
        Dict with:
            lgbm_features: dict of 34 feature values
            lstm_window: numpy array (50, 35) ready for model
            contracts: DataFrame for scoring engine
            spot: current price
            error: str if something failed
    """
    try:
        logger.info(f"Fetching live data for {ticker}...")
        data = _fetch_ticker_data(ticker, days=400)
        prices = data["prices"]
        fundamentals = data["fundamentals"]
        options = data["options"]

        if prices.empty or len(prices) < SEQ_LEN:
            return {"error": f"Insufficient price data for {ticker} ({len(prices)} days)"}

        spot = float(prices["close"].iloc[-1])
        fred = _fetch_fred()

        # ── LGBM features ──
        lgbm_feats = _compute_lgbm_features(prices, fundamentals, options)

        # ── LSTM features (50-day window) ──
        lstm_rows = []
        start_idx = max(0, len(prices) - SEQ_LEN - 200)
        for i in range(start_idx, len(prices)):
            row = _compute_lstm_daily_row(prices.iloc[i], prices, i, fundamentals, fred)
            lstm_rows.append(row)

        lstm_df = pd.DataFrame(lstm_rows)

        # Scale using saved scaler
        lstm_window = None
        try:
            import joblib
            scaler_path = DATA_DIR / "lstm_scaler.joblib"
            if scaler_path.exists():
                scaler = joblib.load(scaler_path)
                # Ensure all expected columns exist
                for col in LSTM_FEATURES:
                    if col not in lstm_df.columns:
                        lstm_df[col] = 0.0
                scaled = scaler.transform(lstm_df[LSTM_FEATURES].fillna(0))
                # Take last 50 rows as the window
                if len(scaled) >= SEQ_LEN:
                    lstm_window = scaled[-SEQ_LEN:].astype(np.float32)
            else:
                logger.warning("LSTM scaler not found — LSTM live inference unavailable")
        except Exception as e:
            logger.warning(f"LSTM scaling failed: {e}")

        # ── Contracts for scoring ──
        contracts = _build_contracts(ticker, options, spot)

        logger.info(f"Live data ready for {ticker}: lgbm={len(lgbm_feats)} feats, "
                     f"lstm={'50x35' if lstm_window is not None else 'unavailable'}, "
                     f"contracts={len(contracts)}")

        return {
            "lgbm_features": lgbm_feats,
            "lstm_window": lstm_window,
            "contracts": contracts,
            "spot": spot,
        }

    except Exception as e:
        logger.error(f"Live data fetch failed for {ticker}: {e}")
        return {"error": f"Live data failed: {e}"}
