"""
LangGraph DAG for scoring inference — Graph 2.

Runs all 6 strategies for a given ticker and date:
  - Baseline: OTM10 always, no model
  - Argmax: model top pick, moneyness-only returns (no scoring engine)
  - Risk-Adjusted: P(bucket) × E[return|bucket], expanding averages, moneyness-only
  - Conservative/Balanced/Aggressive: full scoring engine (TC + delta-hedge + confidence),
    maturity-aware returns

Graph topology:
    START → load_data → [baseline, argmax, risk_adjusted, preset_strategies] → aggregate → END

Separate from the inference graph. Called via its own endpoint after
model predictions render.
"""

import numpy as np
import pandas as pd
from typing import TypedDict
from pathlib import Path
from langgraph.graph import StateGraph, START, END

from src.utils import create_logger

logger = create_logger("scoring_graph")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = _PROJECT_ROOT / "data" / "processed"
SRC_DATA_DIR = _PROJECT_ROOT / "src" / "data"

MONEYNESS_MAP = {0: "ATM", 1: "OTM5", 2: "OTM10"}


class ScoringState(TypedDict):
    ticker: str
    date: str
    month_data: dict        # serialized month rows from feature store
    bucket_returns: dict    # serialized bucket returns for this month
    period: str             # year_month period string
    baseline_result: dict
    argmax_result: dict
    risk_adjusted_result: dict
    preset_results: dict
    combined: dict


# ── Data loading (cached singletons) ───────────────────────────────────

_feature_store_cache = None
_bucket_returns_cache = None


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


def _get_bucket_returns():
    """Load per-bucket realized returns (cached)."""
    global _bucket_returns_cache
    if _bucket_returns_cache is not None:
        return _bucket_returns_cache

    br_path = SRC_DATA_DIR / "bucket_returns.parquet"
    if not br_path.exists():
        logger.warning("bucket_returns.parquet not found")
        return pd.DataFrame()

    br = pd.read_parquet(br_path)
    br["year_month"] = br["year_month"].astype("period[M]")
    _bucket_returns_cache = br
    return br


# ── Nodes ──────────────────────────────────────────────────────────────

async def load_data_node(state: ScoringState) -> dict:
    """Load feature store and bucket returns for the requested month."""
    dt = pd.Timestamp(state["date"])
    period = dt.to_period("M")

    fs = _get_feature_store()
    br = _get_bucket_returns()

    # Get all tickers for this month
    month_data = fs[fs["year_month"] == period]
    if month_data.empty:
        # Snap to nearest month
        available = fs["year_month"].unique()
        diffs = [abs((p - period).n) for p in available]
        nearest = available[pd.Series(diffs).argmin()]
        month_data = fs[fs["year_month"] == nearest]
        period = nearest

    month_br = br[br["year_month"] == period] if not br.empty else pd.DataFrame()

    return {
        "month_data": month_data.to_dict("records"),
        "bucket_returns": month_br.to_dict("records"),
        "period": str(period),
    }


async def baseline_node(state: ScoringState) -> dict:
    """Baseline: OTM10 on all tickers, equal weight. No model."""
    br_rows = pd.DataFrame(state["bucket_returns"])
    if br_rows.empty:
        return {"baseline_result": {"strategy": "Baseline (OTM10)", "return": 0.0, "bucket": "OTM10_SHORT", "n_tickers": 0}}

    returns = br_rows["return_OTM10"].dropna()
    avg_ret = float(returns.mean()) if not returns.empty else 0.0

    return {"baseline_result": {
        "strategy": "Baseline (OTM10)",
        "return": round(avg_ret, 6),
        "bucket": "OTM10_SHORT",
        "n_tickers": len(returns),
        "description": "Sell OTM10 short-dated on all tickers, equal weight. No model.",
    }}


async def argmax_node(state: ScoringState) -> dict:
    """Argmax: model's top pick per ticker, all tickers, equal weight.

    Uses moneyness-only returns — no scoring engine, no maturity dimension.
    """
    month_data = pd.DataFrame(state["month_data"])
    br_rows = pd.DataFrame(state["bucket_returns"])

    if month_data.empty or br_rows.empty:
        return {"argmax_result": {"strategy": "Argmax", "return": 0.0, "n_tickers": 0}}

    returns = []
    picks = []
    for _, row in month_data.iterrows():
        moneyness = row.get("model_moneyness", "OTM10")
        col = f"return_{moneyness}"
        ticker_br = br_rows[br_rows["symbol"] == row["symbol"]]
        if ticker_br.empty or col not in ticker_br.columns:
            continue
        val = ticker_br[col].iloc[0]
        if not pd.isna(val):
            returns.append(float(val))
            picks.append({"ticker": row["symbol"], "bucket": moneyness, "return": round(float(val), 6)})

    avg_ret = float(np.mean(returns)) if returns else 0.0

    return {"argmax_result": {
        "strategy": "Argmax",
        "return": round(avg_ret, 6),
        "n_tickers": len(returns),
        "picks": picks,
        "description": "Model top pick per ticker, moneyness-only returns, equal weight.",
    }}


async def risk_adjusted_node(state: ScoringState) -> dict:
    """Risk-Adjusted: P(bucket) x E[return|bucket], expanding historical averages.

    Uses moneyness-only returns — no scoring engine, no maturity dimension.
    """
    month_data = pd.DataFrame(state["month_data"])
    br_rows = pd.DataFrame(state["bucket_returns"])
    period = pd.Period(state["period"])

    if month_data.empty or br_rows.empty:
        return {"risk_adjusted_result": {"strategy": "Risk-Adjusted", "return": 0.0, "n_tickers": 0}}

    # Compute expanding average returns up to (but not including) this month
    full_br = _get_bucket_returns()
    historical = full_br[full_br["year_month"] < period]
    avg_returns = {}
    for cls in ["ATM", "OTM5", "OTM10"]:
        col = f"return_{cls}"
        if col in historical.columns:
            vals = historical[col].dropna()
            avg_returns[cls] = float(vals.mean()) if not vals.empty else 0.0
        else:
            avg_returns[cls] = 0.0

    returns = []
    picks = []
    for _, row in month_data.iterrows():
        # Score each bucket: P(bucket) x E[return|bucket]
        scores = {}
        for cls in ["ATM", "OTM5", "OTM10"]:
            prob = row.get(f"prob_{cls}", 0.0)
            expected = avg_returns.get(cls, 0.0)
            scores[cls] = prob * expected

        best_bucket = max(scores, key=scores.get)
        col = f"return_{best_bucket}"
        ticker_br = br_rows[br_rows["symbol"] == row["symbol"]]
        if ticker_br.empty or col not in ticker_br.columns:
            continue
        val = ticker_br[col].iloc[0]
        if not pd.isna(val):
            returns.append(float(val))
            picks.append({
                "ticker": row["symbol"], "bucket": best_bucket,
                "return": round(float(val), 6),
                "scores": {k: round(v, 6) for k, v in scores.items()},
            })

    avg_ret = float(np.mean(returns)) if returns else 0.0

    return {"risk_adjusted_result": {
        "strategy": "Risk-Adjusted",
        "return": round(avg_ret, 6),
        "n_tickers": len(returns),
        "picks": picks,
        "avg_historical_returns": {k: round(v, 6) for k, v in avg_returns.items()},
        "description": "P(bucket) x E[return|bucket], expanding averages, equal weight.",
    }}


async def preset_strategies_node(state: ScoringState) -> dict:
    """Run Conservative, Balanced, Aggressive presets via the scoring engine.

    Uses full scoring (TC + delta-hedge + confidence) and maturity-aware returns.
    """
    from src.inference.scoring import score_month, allocate

    month_data = pd.DataFrame(state["month_data"])
    br_rows = pd.DataFrame(state["bucket_returns"])

    if month_data.empty:
        return {"preset_results": {}}

    results = {}
    for preset in ["conservative", "balanced", "aggressive"]:
        try:
            scored = score_month(month_data, prev_buckets={}, preset=preset)
            allocated = allocate(scored, budget=100_000, preset=preset)

            # Compute return from maturity-aware columns
            returns = []
            positions = []
            for _, row in allocated.iterrows():
                bucket = row["model_bucket"]
                col = f"return_{bucket}"
                ticker_br = br_rows[br_rows["symbol"] == row["symbol"]]
                if ticker_br.empty or col not in ticker_br.columns:
                    # Fall back to moneyness-only
                    moneyness = bucket.split("_")[0]
                    col = f"return_{moneyness}"
                    if ticker_br.empty or col not in ticker_br.columns:
                        continue
                val = ticker_br[col].iloc[0] if not ticker_br.empty else None
                if val is not None and not pd.isna(val):
                    returns.append(float(val))
                    positions.append({
                        "ticker": row["symbol"],
                        "bucket": bucket,
                        "composite_score": round(float(row["composite_score"]), 4),
                        "return": round(float(val), 6),
                    })

            avg_ret = float(np.mean(returns)) if returns else 0.0
            results[preset] = {
                "strategy": preset.title(),
                "return": round(avg_ret, 6),
                "n_positions": len(positions),
                "positions": positions,
            }
        except Exception as e:
            logger.error(f"Preset {preset} failed: {e}")
            results[preset] = {"strategy": preset.title(), "return": 0.0, "error": str(e)}

    return {"preset_results": results}


async def aggregate_scoring_node(state: ScoringState) -> dict:
    """Merge all strategy results into a single response."""
    return {"combined": {
        "ticker": state["ticker"],
        "date": state["date"],
        "period": state["period"],
        "baseline": state["baseline_result"],
        "argmax": state["argmax_result"],
        "risk_adjusted": state["risk_adjusted_result"],
        "presets": state["preset_results"],
    }}


# ── Graph builder + invoker ────────────────────────────────────────────

def _build_scoring_graph() -> StateGraph:
    """Construct the scoring DAG."""
    g = StateGraph(ScoringState)
    g.add_node("load_data", load_data_node)
    g.add_node("baseline", baseline_node)
    g.add_node("argmax", argmax_node)
    g.add_node("risk_adjusted", risk_adjusted_node)
    g.add_node("preset_strategies", preset_strategies_node)
    g.add_node("aggregate", aggregate_scoring_node)

    g.add_edge(START, "load_data")
    g.add_edge("load_data", "baseline")
    g.add_edge("load_data", "argmax")
    g.add_edge("load_data", "risk_adjusted")
    g.add_edge("load_data", "preset_strategies")
    g.add_edge("baseline", "aggregate")
    g.add_edge("argmax", "aggregate")
    g.add_edge("risk_adjusted", "aggregate")
    g.add_edge("preset_strategies", "aggregate")
    g.add_edge("aggregate", END)

    return g.compile()


async def invoke_scoring_graph(ticker: str, date: str) -> dict:
    """Compile and invoke the scoring graph.

    Args:
        ticker: Stock symbol (used for context, scoring runs all tickers).
        date: Date string (YYYY-MM-DD).

    Returns:
        Combined dict with all 6 strategy results.
    """
    graph = _build_scoring_graph()
    result = await graph.ainvoke({
        "ticker": ticker,
        "date": date,
        "month_data": {},
        "bucket_returns": {},
        "period": "",
        "baseline_result": {},
        "argmax_result": {},
        "risk_adjusted_result": {},
        "preset_results": {},
        "combined": {},
    })
    return result["combined"]
