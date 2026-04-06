"""
LangGraph DAG for market context — Graph 3.

Builds regime and market context for a given ticker and date:
  - Trailing price data from daily_clean.parquet
  - LGBM feature store row (34 features: IV, technicals, fundamentals)
  - Model track record for this ticker
  - Capstone insights (static, loaded once)

Graph topology:
    START → [price_context, feature_context, model_track_record] → aggregate → END

Separate from inference and scoring graphs. Called via its own endpoint
after model predictions render. Can run in parallel with scoring graph.
"""

import numpy as np
import pandas as pd
from typing import TypedDict
from pathlib import Path
from langgraph.graph import StateGraph, START, END

from src.utils import create_logger

logger = create_logger("context_graph")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = _PROJECT_ROOT / "data" / "processed"
INSIGHTS_PATH = Path(__file__).resolve().parent / "capstone_insights.md"

# ── Cached singletons ──────────────────────────────────────────────────

_daily_cache = None
_feature_store_cache = None
_insights_cache = None


def _get_daily():
    """Load daily price data (cached)."""
    global _daily_cache
    if _daily_cache is not None:
        return _daily_cache
    df = pd.read_parquet(DATA_DIR / "daily_clean.parquet")
    df["date"] = pd.to_datetime(df["date"])
    _daily_cache = df
    return df


def _get_feature_store():
    """Load LGBM feature store (cached)."""
    global _feature_store_cache
    if _feature_store_cache is not None:
        return _feature_store_cache
    fs = pd.read_parquet(DATA_DIR / "feature_store.parquet")
    fs["decision_date"] = pd.to_datetime(fs["decision_date"])
    fs["year_month"] = fs["decision_date"].dt.to_period("M")
    _feature_store_cache = fs
    return fs


def _get_insights():
    """Load capstone insights markdown (cached)."""
    global _insights_cache
    if _insights_cache is not None:
        return _insights_cache
    if INSIGHTS_PATH.exists():
        _insights_cache = INSIGHTS_PATH.read_text()
    else:
        _insights_cache = ""
    return _insights_cache


class ContextState(TypedDict):
    ticker: str
    date: str
    price_context: dict
    feature_context: dict
    track_record: dict
    combined: dict


# ── Nodes ──────────────────────────────────────────────────────────────

async def price_context_node(state: ContextState) -> dict:
    """Pull trailing price data for regime understanding.

    Returns 60 trading days of OHLCV + computed metrics:
    trend direction, volatility regime, recent drawdown.
    """
    try:
        daily = _get_daily()
        ticker = state["ticker"].upper()
        dt = pd.Timestamp(state["date"])

        ticker_data = daily[daily["symbol"] == ticker].sort_values("date")
        trailing = ticker_data[ticker_data["date"] <= dt].tail(60)

        if trailing.empty:
            return {"price_context": {"error": f"No daily data for {ticker}"}}

        close = trailing["adjusted_close"].values
        n = len(close)

        # Trend: simple 20d vs 60d SMA comparison
        sma_20 = float(close[-20:].mean()) if n >= 20 else float(close.mean())
        sma_60 = float(close.mean())
        trend = "bullish" if sma_20 > sma_60 else "bearish"

        # Volatility: annualized from daily returns
        if n >= 2:
            returns = np.diff(close) / close[:-1]
            vol_20d = float(np.std(returns[-20:]) * np.sqrt(252)) if n >= 21 else float(np.std(returns) * np.sqrt(252))
            vol_60d = float(np.std(returns) * np.sqrt(252))
        else:
            vol_20d = vol_60d = 0.0

        # Regime classification
        if vol_20d > 0.40:
            vol_regime = "high_volatility"
        elif vol_20d > 0.20:
            vol_regime = "normal"
        else:
            vol_regime = "low_volatility"

        # Recent drawdown from peak
        peak = float(close.max())
        current = float(close[-1])
        drawdown = (current - peak) / peak if peak > 0 else 0.0

        # Price range
        period_return = (float(close[-1]) / float(close[0]) - 1) if close[0] > 0 else 0.0

        return {"price_context": {
            "ticker": ticker,
            "current_price": round(current, 2),
            "period_days": n,
            "sma_20": round(sma_20, 2),
            "sma_60": round(sma_60, 2),
            "trend": trend,
            "vol_20d": round(vol_20d, 4),
            "vol_60d": round(vol_60d, 4),
            "vol_regime": vol_regime,
            "drawdown_from_peak": round(drawdown, 4),
            "period_return": round(period_return, 4),
        }}
    except Exception as e:
        logger.error(f"price_context_node failed: {e}")
        return {"price_context": {"error": str(e)}}


async def feature_context_node(state: ContextState) -> dict:
    """Pull LGBM feature store row for this ticker/date.

    Returns the 34 model features + IV metrics for regime interpretation.
    """
    try:
        fs = _get_feature_store()
        ticker = state["ticker"].upper()
        dt = pd.Timestamp(state["date"])
        period = dt.to_period("M")

        rows = fs[fs["symbol"] == ticker]
        if rows.empty:
            return {"feature_context": {"error": f"No feature data for {ticker}"}}

        # Match month or snap
        matches = rows[rows["year_month"] == period]
        if matches.empty:
            available = rows["year_month"].unique()
            diffs = [abs((p - period).n) for p in available]
            nearest = available[pd.Series(diffs).argmin()]
            matches = rows[rows["year_month"] == nearest]

        row = matches.iloc[0]

        # Key IV features for regime context
        iv_features = {}
        for col in ["iv_mean", "iv_median", "iv_skew", "iv_short_mean",
                     "iv_long_mean", "iv_term_structure", "iv_rank", "iv_change"]:
            if col in row.index:
                iv_features[col] = round(float(row[col]), 4) if pd.notna(row[col]) else None

        # Key technical features
        tech_features = {}
        for col in ["return_1m", "return_3m", "momentum_21d", "vol_21d",
                     "rsi_14", "natr", "bb_width", "macd_signal"]:
            if col in row.index:
                tech_features[col] = round(float(row[col]), 4) if pd.notna(row[col]) else None

        # Key fundamental features
        fund_features = {}
        for col in ["gross_margin", "revenue_growth", "roe", "debt_to_equity",
                     "pe_ratio", "fcf_yield"]:
            if col in row.index:
                fund_features[col] = round(float(row[col]), 4) if pd.notna(row[col]) else None

        # Model prediction context
        model_context = {
            "model_moneyness": row.get("model_moneyness", "?"),
            "model_maturity": row.get("model_maturity", "?"),
            "model_bucket": row.get("model_bucket", "?"),
            "model_confidence": round(float(row.get("model_confidence", 0)), 4),
            "prob_ATM": round(float(row.get("prob_ATM", 0)), 4),
            "prob_OTM5": round(float(row.get("prob_OTM5", 0)), 4),
            "prob_OTM10": round(float(row.get("prob_OTM10", 0)), 4),
        }

        return {"feature_context": {
            "ticker": ticker,
            "month": str(matches.iloc[0]["year_month"]),
            "iv": iv_features,
            "technical": tech_features,
            "fundamental": fund_features,
            "model": model_context,
        }}
    except Exception as e:
        logger.error(f"feature_context_node failed: {e}")
        return {"feature_context": {"error": str(e)}}


async def model_track_record_node(state: ContextState) -> dict:
    """Compute model accuracy track record for this ticker and nearby periods."""
    try:
        fs = _get_feature_store()
        ticker = state["ticker"].upper()

        rows = fs[fs["symbol"] == ticker]
        if rows.empty:
            return {"track_record": {"error": f"No data for {ticker}"}}

        n_total = len(rows)
        n_correct = int(rows["model_correct"].sum())
        accuracy = round(n_correct / n_total, 4) if n_total > 0 else 0.0

        # Per-class accuracy for this ticker
        per_class = {}
        for cls in ["ATM", "OTM5", "OTM10"]:
            cls_rows = rows[rows["true_moneyness"] == cls]
            if not cls_rows.empty:
                per_class[cls] = {
                    "n": len(cls_rows),
                    "accuracy": round(float(cls_rows["model_correct"].mean()), 4),
                }

        # Recent accuracy (last 12 months)
        recent = rows.nlargest(12, "decision_date")
        recent_acc = round(float(recent["model_correct"].mean()), 4) if not recent.empty else 0.0

        # Avg confidence when correct vs incorrect
        correct = rows[rows["model_correct"]]
        incorrect = rows[~rows["model_correct"]]

        return {"track_record": {
            "ticker": ticker,
            "total_predictions": n_total,
            "overall_accuracy": accuracy,
            "recent_12m_accuracy": recent_acc,
            "per_class": per_class,
            "avg_confidence_when_correct": round(float(correct["model_confidence"].mean()), 4) if not correct.empty else 0.0,
            "avg_confidence_when_incorrect": round(float(incorrect["model_confidence"].mean()), 4) if not incorrect.empty else 0.0,
        }}
    except Exception as e:
        logger.error(f"model_track_record_node failed: {e}")
        return {"track_record": {"error": str(e)}}


async def aggregate_context_node(state: ContextState) -> dict:
    """Merge all context signals into a single response."""
    insights = _get_insights()

    return {"combined": {
        "ticker": state["ticker"],
        "date": state["date"],
        "price": state["price_context"],
        "features": state["feature_context"],
        "track_record": state["track_record"],
        "insights_available": len(insights) > 0,
    }}


# ── Graph builder + invoker ────────────────────────────────────────────

def _build_context_graph() -> StateGraph:
    """Construct the context DAG."""
    g = StateGraph(ContextState)
    g.add_node("price_context", price_context_node)
    g.add_node("feature_context", feature_context_node)
    g.add_node("model_track_record", model_track_record_node)
    g.add_node("aggregate", aggregate_context_node)

    g.add_edge(START, "price_context")
    g.add_edge(START, "feature_context")
    g.add_edge(START, "model_track_record")
    g.add_edge("price_context", "aggregate")
    g.add_edge("feature_context", "aggregate")
    g.add_edge("model_track_record", "aggregate")
    g.add_edge("aggregate", END)

    return g.compile()


async def invoke_context_graph(ticker: str, date: str) -> dict:
    """Compile and invoke the context graph.

    Args:
        ticker: Stock symbol.
        date: Date string (YYYY-MM-DD).

    Returns:
        Combined dict with price, feature, and track record context.
    """
    graph = _build_context_graph()
    result = await graph.ainvoke({
        "ticker": ticker,
        "date": date,
        "price_context": {},
        "feature_context": {},
        "track_record": {},
        "combined": {},
    })
    return result["combined"]
