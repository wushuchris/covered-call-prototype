"""
LangGraph DAG for scoring inference, Graph 2.

Runs strategies for both LGBM and LSTM-CNN models:
  - Baseline: OTM10 always, no model (shared)
  - Per model (LGBM + LSTM separately):
    - Argmax: model top pick, moneyness-only returns
    - Risk-Adjusted: P(bucket) x E[return|bucket], expanding averages
    - Conservative: full scoring engine (TC + delta-hedge + confidence)

Graph topology:
    START -> load_data -> [baseline, lgbm_strategies, lstm_strategies] -> aggregate -> END
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


class ScoringState(TypedDict):
    ticker: str
    date: str
    lgbm_month_data: dict
    lstm_month_data: dict
    bucket_returns: dict
    period: str
    baseline_result: dict
    lgbm_result: dict
    lstm_result: dict
    combined: dict


# -- Data loading (cached singletons) --

_lgbm_fs_cache = None
_lstm_fs_cache = None
_bucket_returns_cache = None


def _get_lgbm_feature_store():
    """Load LGBM feature store (cached)."""
    global _lgbm_fs_cache
    if _lgbm_fs_cache is not None:
        return _lgbm_fs_cache
    fs = pd.read_parquet(DATA_DIR / "feature_store.parquet")
    fs["decision_date"] = pd.to_datetime(fs["decision_date"])
    fs["year_month"] = fs["decision_date"].dt.to_period("M")
    _lgbm_fs_cache = fs
    return fs


def _get_lstm_monthly():
    """Load LSTM monthly predictions (cached)."""
    global _lstm_fs_cache
    if _lstm_fs_cache is not None:
        return _lstm_fs_cache
    from src.inference.lstm_model import get_monthly_predictions
    _lstm_fs_cache = get_monthly_predictions()
    return _lstm_fs_cache


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


# -- Shared helpers --

def _compute_argmax(month_data, br_rows):
    """Argmax: model top pick per ticker, moneyness-only returns, equal weight."""
    returns, picks = [], []
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
    return {"strategy": "Argmax", "return": round(avg_ret, 6), "n_tickers": len(returns), "picks": picks}


def _compute_risk_adjusted(month_data, br_rows, period):
    """Risk-Adjusted: P(bucket) x E[return|bucket], expanding averages."""
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

    returns, picks = [], []
    for _, row in month_data.iterrows():
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
            picks.append({"ticker": row["symbol"], "bucket": best_bucket, "return": round(float(val), 6)})

    avg_ret = float(np.mean(returns)) if returns else 0.0
    return {"strategy": "Risk-Adjusted", "return": round(avg_ret, 6), "n_tickers": len(returns), "picks": picks}


def _compute_conservative(month_data, br_rows):
    """Conservative: full scoring engine (TC + delta-hedge + confidence)."""
    from src.inference.scoring import score_month, allocate
    try:
        scored = score_month(month_data, prev_buckets={}, preset="conservative")
        allocated = allocate(scored, budget=100_000, preset="conservative")

        returns, positions = [], []
        for _, row in allocated.iterrows():
            bucket = row["model_bucket"]
            col = f"return_{bucket}"
            ticker_br = br_rows[br_rows["symbol"] == row["symbol"]]
            if ticker_br.empty or col not in ticker_br.columns:
                moneyness = bucket.split("_")[0]
                col = f"return_{moneyness}"
                if ticker_br.empty or col not in ticker_br.columns:
                    continue
            val = ticker_br[col].iloc[0] if not ticker_br.empty else None
            if val is not None and not pd.isna(val):
                returns.append(float(val))
                positions.append({"ticker": row["symbol"], "bucket": bucket, "return": round(float(val), 6)})

        avg_ret = float(np.mean(returns)) if returns else 0.0
        return {"strategy": "Conservative", "return": round(avg_ret, 6), "n_positions": len(positions)}
    except Exception as e:
        logger.error(f"Conservative scoring failed: {e}")
        return {"strategy": "Conservative", "return": 0.0, "error": str(e)}


# -- Nodes --

async def load_data_node(state: ScoringState) -> dict:
    """Load both feature stores and bucket returns for the requested month."""
    dt = pd.Timestamp(state["date"])
    period = dt.to_period("M")

    lgbm_fs = _get_lgbm_feature_store()
    lstm_fs = _get_lstm_monthly()
    br = _get_bucket_returns()

    # LGBM month data (snap to nearest if needed)
    lgbm_month = lgbm_fs[lgbm_fs["year_month"] == period]
    if lgbm_month.empty:
        available = lgbm_fs["year_month"].unique()
        diffs = [abs((p - period).n) for p in available]
        nearest = available[pd.Series(diffs).argmin()]
        lgbm_month = lgbm_fs[lgbm_fs["year_month"] == nearest]
        period = nearest

    # LSTM month data (snap to nearest if needed)
    lstm_month = lstm_fs[lstm_fs["year_month"] == period] if not lstm_fs.empty else pd.DataFrame()
    if lstm_month.empty and not lstm_fs.empty:
        available = lstm_fs["year_month"].unique()
        diffs = [abs((p - period).n) for p in available]
        nearest_lstm = available[pd.Series(diffs).argmin()]
        lstm_month = lstm_fs[lstm_fs["year_month"] == nearest_lstm]

    month_br = br[br["year_month"] == period] if not br.empty else pd.DataFrame()

    return {
        "lgbm_month_data": lgbm_month.to_dict("records"),
        "lstm_month_data": lstm_month.to_dict("records"),
        "bucket_returns": month_br.to_dict("records"),
        "period": str(period),
    }


async def baseline_node(state: ScoringState) -> dict:
    """Baseline: OTM10 on all tickers, equal weight. No model."""
    br_rows = pd.DataFrame(state["bucket_returns"])
    if br_rows.empty:
        return {"baseline_result": {"strategy": "Baseline (OTM10)", "return": 0.0, "n_tickers": 0}}
    returns = br_rows["return_OTM10"].dropna()
    avg_ret = float(returns.mean()) if not returns.empty else 0.0
    return {"baseline_result": {
        "strategy": "Baseline (OTM10)",
        "return": round(avg_ret, 6),
        "n_tickers": len(returns),
    }}


async def lgbm_strategies_node(state: ScoringState) -> dict:
    """Run all LGBM strategies: Argmax, Risk-Adjusted, Conservative."""
    month_data = pd.DataFrame(state["lgbm_month_data"])
    br_rows = pd.DataFrame(state["bucket_returns"])
    period = pd.Period(state["period"])

    if month_data.empty or br_rows.empty:
        empty = {"strategy": "N/A", "return": 0.0, "n_tickers": 0}
        return {"lgbm_result": {"argmax": empty, "risk_adjusted": empty, "conservative": empty}}

    return {"lgbm_result": {
        "argmax": _compute_argmax(month_data, br_rows),
        "risk_adjusted": _compute_risk_adjusted(month_data, br_rows, period),
        "conservative": _compute_conservative(month_data, br_rows),
    }}


async def lstm_strategies_node(state: ScoringState) -> dict:
    """Run all LSTM strategies: Argmax, Risk-Adjusted, Conservative."""
    month_data = pd.DataFrame(state["lstm_month_data"])
    br_rows = pd.DataFrame(state["bucket_returns"])
    period = pd.Period(state["period"])

    if month_data.empty or br_rows.empty:
        empty = {"strategy": "N/A", "return": 0.0, "n_tickers": 0}
        return {"lstm_result": {"argmax": empty, "risk_adjusted": empty, "conservative": empty}}

    return {"lstm_result": {
        "argmax": _compute_argmax(month_data, br_rows),
        "risk_adjusted": _compute_risk_adjusted(month_data, br_rows, period),
        "conservative": _compute_conservative(month_data, br_rows),
    }}


async def aggregate_scoring_node(state: ScoringState) -> dict:
    """Merge both model results + baseline into a single response."""
    return {"combined": {
        "ticker": state["ticker"],
        "date": state["date"],
        "period": state["period"],
        "baseline": state["baseline_result"],
        "lgbm": state["lgbm_result"],
        "lstm": state["lstm_result"],
    }}


# -- Graph builder + invoker --

def _build_scoring_graph() -> StateGraph:
    """Construct the dual-model scoring DAG."""
    g = StateGraph(ScoringState)
    g.add_node("load_data", load_data_node)
    g.add_node("baseline", baseline_node)
    g.add_node("lgbm_strategies", lgbm_strategies_node)
    g.add_node("lstm_strategies", lstm_strategies_node)
    g.add_node("aggregate", aggregate_scoring_node)

    g.add_edge(START, "load_data")
    g.add_edge("load_data", "baseline")
    g.add_edge("load_data", "lgbm_strategies")
    g.add_edge("load_data", "lstm_strategies")
    g.add_edge("baseline", "aggregate")
    g.add_edge("lgbm_strategies", "aggregate")
    g.add_edge("lstm_strategies", "aggregate")
    g.add_edge("aggregate", END)

    return g.compile()


async def invoke_scoring_graph(ticker: str, date: str) -> dict:
    """Compile and invoke the scoring graph.

    Args:
        ticker: Stock symbol (for context, scoring runs all tickers).
        date: Date string (YYYY-MM-DD).

    Returns:
        Combined dict with baseline + LGBM strategies + LSTM strategies.
    """
    graph = _build_scoring_graph()
    result = await graph.ainvoke({
        "ticker": ticker,
        "date": date,
        "lgbm_month_data": {},
        "lstm_month_data": {},
        "bucket_returns": {},
        "period": "",
        "baseline_result": {},
        "lgbm_result": {},
        "lstm_result": {},
        "combined": {},
    })
    return result["combined"]
