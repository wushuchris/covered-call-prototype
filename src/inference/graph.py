"""
LangGraph DAG for dual-model inference.

Spawned per request. Runs LGBM (3-class) and LSTM-CNN (7-class) in parallel,
then aggregates into a single response dict.

Graph topology:
    START → validate → [lgbm, lstm] (parallel) → aggregate → END

State is minimal: input (ticker, date) + output dicts per model + combined.
"""

from typing import TypedDict
from langgraph.graph import StateGraph, START, END

from src.inference.inference_utils import validate_ticker, validate_date
from src.utils import create_logger

logger = create_logger("graph")

UNIVERSE = ["AAPL", "AMZN", "AVGO", "GOOG", "GOOGL", "META", "MSFT", "NVDA", "TSLA", "WMT"]


class GraphState(TypedDict):
    ticker: str
    date: str
    lgbm_result: dict
    lstm_result: dict
    combined: dict


# ── Nodes ────────────────────────────────────────────────────────────────

async def validate_node(state: GraphState) -> dict:
    """Validate ticker and date. Raises on failure (caught by caller)."""
    ticker = state["ticker"]
    date = state["date"]

    if not validate_ticker(ticker, UNIVERSE):
        raise ValueError(f"Ticker '{ticker}' not in universe: {UNIVERSE}")
    if not validate_date(date):
        raise ValueError(f"Invalid date format: '{date}'. Expected YYYY-MM-DD.")

    return {}


async def lgbm_node(state: GraphState) -> dict:
    """Run LGBM 3-class prediction (row lookup from feature store)."""
    from src.inference.model import predict_bucket
    result = predict_bucket(state["ticker"], state["date"])
    return {"lgbm_result": result}


async def lstm_node(state: GraphState) -> dict:
    """Run LSTM-CNN 7-class prediction (row lookup from feature store)."""
    from src.inference.lstm_model import predict_bucket
    result = predict_bucket(state["ticker"], state["date"])
    return {"lstm_result": result}


async def aggregate_node(state: GraphState) -> dict:
    """Merge both model results into a single response."""
    lgbm = state["lgbm_result"]
    lstm = state["lstm_result"]

    combined = {
        "ticker": state["ticker"],
        "date": state["date"],
        # LGBM fields (kept flat for backward compat)
        "month": lgbm.get("month", "—"),
        "snapped": lgbm.get("snapped", False) or lstm.get("snapped", False),
        "model_bucket": lgbm.get("model_bucket", "—"),
        "model_confidence": lgbm.get("model_confidence", 0),
        "baseline": lgbm.get("baseline", "OTM10_SHORT"),
        "sample_type": lgbm.get("sample_type", "—"),
        # LSTM fields (namespaced)
        "lstm_prediction": lstm.get("predicted_class", "—"),
        "lstm_confidence": lstm.get("confidence", 0),
        "lstm_sample_type": lstm.get("sample_type", "—"),
        # Full sub-dicts for anything the UI needs
        "lgbm": lgbm,
        "lstm": lstm,
    }
    return {"combined": combined}


# ── Graph builder + invoker ──────────────────────────────────────────────

def _build_graph() -> StateGraph:
    """Construct the inference DAG. Called per request."""
    g = StateGraph(GraphState)
    g.add_node("validate", validate_node)
    g.add_node("lgbm", lgbm_node)
    g.add_node("lstm", lstm_node)
    g.add_node("aggregate", aggregate_node)

    g.add_edge(START, "validate")
    g.add_edge("validate", "lgbm")
    g.add_edge("validate", "lstm")
    g.add_edge("lgbm", "aggregate")
    g.add_edge("lstm", "aggregate")
    g.add_edge("aggregate", END)

    return g.compile()


async def invoke_graph(ticker: str, date: str) -> dict:
    """Compile graph, invoke with fresh state, return combined result.

    Args:
        ticker: Stock symbol.
        date: Date string (YYYY-MM-DD).

    Returns:
        Combined dict with both model predictions.
    """
    graph = _build_graph()
    result = await graph.ainvoke({
        "ticker": ticker,
        "date": date,
        "lgbm_result": {},
        "lstm_result": {},
        "combined": {},
    })
    return result["combined"]
