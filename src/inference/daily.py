# Daily inference — orchestrates the LangGraph DAG and appends chart data.
#
# run_daily_inference: single ticker → graph → chart → response
# run_batch_inference: all tickers → graph per ticker → summary stats

from src.inference.graph import invoke_graph
from src.inference.chart import build_candlestick_data

UNIVERSE = ["AAPL", "AMZN", "AVGO", "GOOG", "GOOGL", "META", "MSFT", "NVDA", "TSLA", "WMT"]


async def run_daily_inference(ticker: str, date: str) -> dict:
    """Run dual-model inference for a single ticker via LangGraph.

    Args:
        ticker: Stock symbol from the user.
        date: Date string (YYYY-MM-DD).

    Returns:
        Dict with combined model results + chart data, or error.
    """
    try:
        result = await invoke_graph(ticker=ticker, date=date)
        result["chart_data"] = build_candlestick_data(ticker=ticker, date=date)
        return result
    except Exception as e:
        return {"error": f"Daily inference failed: {e}"}


async def run_batch_inference(date: str) -> dict:
    """Run dual-model inference for all tickers via LangGraph.

    Args:
        date: Date string (YYYY-MM-DD).

    Returns:
        Dict with per-ticker results and summary stats.
    """
    try:
        results = []
        for ticker in UNIVERSE:
            try:
                result = await invoke_graph(ticker=ticker, date=date)
                result["chart_data"] = build_candlestick_data(ticker=ticker, date=date)
            except Exception as e:
                result = {"error": str(e), "ticker": ticker}
            results.append(result)

        valid = [r for r in results if "error" not in r]
        if not valid:
            return {"error": "No valid predictions for any ticker on this date."}

        top = max(valid, key=lambda r: r.get("model_confidence", 0))
        avg_conf = sum(r.get("model_confidence", 0) for r in valid) / len(valid)

        summary = {
            "date": date,
            "n_tickers": len(valid),
            "n_errors": len(results) - len(valid),
            "top_ticker": top.get("ticker", "?"),
            "top_prediction": top.get("model_bucket", "?"),
            "top_confidence": round(top.get("model_confidence", 0), 4),
            "avg_confidence": round(avg_conf, 4),
            "sample_type": valid[0].get("sample_type", "?"),
            "model": "LGBM 3-Class + LSTM-CNN 7-Class",
        }

        return {"results": results, "summary": summary}

    except Exception as e:
        return {"error": f"Batch inference failed: {e}"}
