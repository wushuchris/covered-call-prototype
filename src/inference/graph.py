"""
LangGraph DAG for dual-model inference.

Spawned per request. Runs LGBM (3-class) and LSTM-CNN (7-class) in parallel,
then aggregates into a single response dict.

Supports two modes:
    - Historical: date within feature store range → row lookup (instant)
    - Live: date beyond feature stores → fetch from yfinance + compute features

Graph topology:
    START → validate → [lgbm, lstm] (parallel) → aggregate → END
"""

import pandas as pd
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
    """Validate ticker and date."""
    ticker = state["ticker"]
    date = state["date"]

    if not validate_ticker(ticker, UNIVERSE):
        raise ValueError(f"Ticker '{ticker}' not in universe: {UNIVERSE}")
    if not validate_date(date):
        raise ValueError(f"Invalid date format: '{date}'. Expected YYYY-MM-DD.")

    return {}


async def lgbm_node(state: GraphState) -> dict:
    """Run LGBM 3-class prediction — checks its own store range."""
    ticker, date = state["ticker"], state["date"]

    from src.inference.model import get_date_range
    lgbm_max = pd.Timestamp(get_date_range()["max_date"])
    if pd.Timestamp(date) > lgbm_max:
        logger.info(f"LGBM live: {date} > {lgbm_max.date()}")
        return _lgbm_live(ticker, date)

    from src.inference.model import predict_bucket
    return {"lgbm_result": predict_bucket(ticker, date)}


async def lstm_node(state: GraphState) -> dict:
    """Run LSTM-CNN 7-class prediction — checks its own store range."""
    ticker, date = state["ticker"], state["date"]

    from src.inference.lstm_model import get_max_date
    lstm_max = pd.Timestamp(get_max_date())
    if pd.Timestamp(date) > lstm_max:
        logger.info(f"LSTM live: {date} > {lstm_max.date()}")
        return _lstm_live(ticker, date)

    from src.inference.lstm_model import predict_bucket
    return {"lstm_result": predict_bucket(ticker, date)}


def _lgbm_live(ticker: str, date: str) -> dict:
    """LGBM prediction from live yfinance data."""
    try:
        from src.inference.live_data import fetch_live, LGBM_FEATURES
        from src.inference.model import _load_model, MONEYNESS_MAP

        live = fetch_live(ticker)
        if "error" in live:
            return {"lgbm_result": {"error": live["error"], "ticker": ticker, "date": date}}

        feats = live["lgbm_features"]
        if not feats:
            return {"lgbm_result": {"error": "Could not compute LGBM features", "ticker": ticker, "date": date}}

        model, _ = _load_model()
        import numpy as np
        X = np.array([[feats.get(f, 0.0) for f in LGBM_FEATURES]])
        pred = model.predict(X)[0]
        proba = model.predict_proba(X)[0]

        # IV-rank maturity rule
        iv_rank = feats.get("iv_rank", 0.5)
        maturity = "SHORT" if iv_rank > 0.5 else "LONG"
        moneyness = MONEYNESS_MAP[pred]

        return {"lgbm_result": {
            "ticker": ticker, "date": date,
            "month": date[:7], "snapped": False,
            "model_bucket": f"{moneyness}_{maturity}",
            "model_moneyness": moneyness,
            "model_maturity": maturity,
            "model_confidence": round(float(proba.max()), 4),
            "baseline": "OTM10_SHORT",
            "sample_type": "Live",
            "is_live": True,
        }}

    except Exception as e:
        logger.error(f"LGBM live failed: {e}")
        return {"lgbm_result": {"error": f"LGBM live failed: {e}", "ticker": ticker, "date": date}}


def _lstm_live(ticker: str, date: str) -> dict:
    """LSTM-CNN prediction from live yfinance data."""
    try:
        from src.inference.live_data import fetch_live, LSTM_FEATURES
        from src.inference.lstm_model import _load_model, CLASS_NAMES, _CLASS_TO_MONEYNESS, _CLASS_TO_MATURITY
        import torch
        import numpy as np

        live = fetch_live(ticker)
        if "error" in live:
            return {"lstm_result": {"error": live["error"], "ticker": ticker, "date": date}}

        window = live.get("lstm_window")
        if window is None:
            return {"lstm_result": {"error": "Could not build LSTM window", "ticker": ticker, "date": date}}

        model, _ = _load_model()
        with torch.no_grad():
            x = torch.from_numpy(window[np.newaxis]).float()
            logits = model(x)
            probs = torch.softmax(logits, dim=-1).numpy()[0]

        pred_id = int(probs.argmax())
        pred_class = CLASS_NAMES[pred_id]

        return {"lstm_result": {
            "ticker": ticker, "date": date,
            "actual_date": date, "snapped": False,
            "model_name": "LSTM-CNN 7-Class",
            "predicted_class": pred_class,
            "confidence": round(float(probs.max()), 4),
            "true_label": "—",
            "correct": False,
            "sample_type": "Live",
            "is_live": True,
            "probabilities": {cls: round(float(probs[i]), 4) for i, cls in enumerate(CLASS_NAMES)},
        }}

    except Exception as e:
        logger.error(f"LSTM live failed: {e}")
        return {"lstm_result": {"error": f"LSTM live failed: {e}", "ticker": ticker, "date": date}}


async def aggregate_node(state: GraphState) -> dict:
    """Merge both model results into a single response."""
    lgbm = state["lgbm_result"]
    lstm = state["lstm_result"]

    # Either model running live triggers the experimental banner
    is_live = lgbm.get("is_live", False) or lstm.get("is_live", False)

    combined = {
        "ticker": state["ticker"],
        "date": state["date"],
        "is_live": is_live,
        # LGBM fields
        "month": lgbm.get("month", "—"),
        "snapped": lgbm.get("snapped", False) or lstm.get("snapped", False),
        "model_bucket": lgbm.get("model_bucket", "—"),
        "model_confidence": lgbm.get("model_confidence", 0),
        "baseline": lgbm.get("baseline", "OTM10_SHORT"),
        "sample_type": lgbm.get("sample_type", "—"),
        # LSTM fields
        "lstm_prediction": lstm.get("predicted_class", "—"),
        "lstm_confidence": lstm.get("confidence", 0),
        "lstm_sample_type": lstm.get("sample_type", "—"),
        # Full sub-dicts
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
    """Compile graph, invoke with fresh state, return combined result."""
    graph = _build_graph()
    result = await graph.ainvoke({
        "ticker": ticker,
        "date": date,
        "lgbm_result": {},
        "lstm_result": {},
        "combined": {},
    })
    return result["combined"]
