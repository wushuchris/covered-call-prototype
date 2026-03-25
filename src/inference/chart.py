"""
OHLC data builder for the inference UI candlestick chart.

Loads daily price data from processed parquets, slices from dataset
start to the user-specified date, and returns a list of dicts
ready for ApexCharts candlestick rendering on the UI side.
"""

import pandas as pd
from pathlib import Path

from src.utils import create_logger

logger = create_logger("chart")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = _PROJECT_ROOT / "data" / "processed"

# Cache the daily dataframe in memory after first load
_daily_df = None


def _load_daily():
    """Load daily_clean.parquet once and cache in module state."""
    global _daily_df
    if _daily_df is not None:
        return _daily_df

    try:
        df = pd.read_parquet(DATA_DIR / "daily_clean.parquet")
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
        _daily_df = df
        logger.info(f"Daily data loaded: {len(df)} rows.")
        return df
    except Exception as e:
        logger.error(f"Failed to load daily data: {e}")
        raise


def build_candlestick_data(ticker: str, date: str, max_days: int = 23) -> list:
    """Build OHLC data for the trailing month up to the given date.

    Limits to ~1 month of trading days by default, matching the
    monthly prediction window.

    Args:
        ticker: Stock symbol (e.g. "AAPL").
        date: End date (inclusive), any pandas-parseable string.
        max_days: Maximum number of trading days to include.

    Returns:
        List of dicts with {"x": iso_date, "y": [open, high, low, close]}
        formatted for ApexCharts candlestick series. Empty list on failure.
    """
    try:
        df = _load_daily()
        end = pd.Timestamp(date)

        mask = (df["symbol"] == ticker.upper()) & (df["date"] <= end)
        subset = df.loc[mask].tail(max_days)

        if subset.empty:
            return []

        # Vectorized — no iterrows
        dates = subset["date"].dt.strftime("%Y-%m-%d").tolist()
        opens = subset["open"].round(2).tolist()
        highs = subset["high"].round(2).tolist()
        lows = subset["low"].round(2).tolist()
        closes = subset["close"].round(2).tolist()

        return [
            {"x": d, "y": [o, h, l, c]}
            for d, o, h, l, c in zip(dates, opens, highs, lows, closes)
        ]

    except Exception as e:
        logger.error(f"Chart data build failed for {ticker} {date}: {e}")
        return []
