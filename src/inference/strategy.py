# Covered call strategy — shared between daily inference (UI) and backtesting.
#
# predict_bucket() from model.py is the single source of truth for decisions.
# This module wraps it for the FastHTML UI via daily.py.
#
# The strategy itself does no feature computation or model loading —
# it asks model.py "what bucket for this ticker on this date?" and acts on the answer.
#
# NautilusTrader was previously scaffolded here but removed — the backtest
# loop runs in plain Python via scoring.py. See nautilus_reference.py for
# the original class if needed in the future.

from src.inference.model import predict_bucket

# 6-class buckets (moneyness x maturity) — matches the model output
BUCKETS_6 = [
    "ATM_SHORT", "ATM_LONG",
    "OTM5_SHORT", "OTM5_LONG",
    "OTM10_SHORT", "OTM10_LONG",
]

BASELINE_BUCKET = "OTM10_SHORT"


# ── Daily inference entry point (called by daily.py → FastHTML UI) ───────────

def simulate_inference(ticker: str, date: str) -> dict:
    """Run model inference for a given ticker and date.

    Delegates entirely to model.predict_bucket(). Returns the full
    result dict including model prediction, ground truth, and comparison.

    Args:
        ticker: Stock symbol.
        date: Date string (YYYY-MM-DD).

    Returns:
        Dict with inference results or error info.
    """
    try:
        return predict_bucket(ticker, date)
    except Exception as e:
        return {"error": f"Inference failed: {e}"}


def baseline_strategy() -> str:
    """Baseline comparison strategy: always sell short-dated 10% OTM calls.

    Returns:
        The baseline bucket label.
    """
    return BASELINE_BUCKET
