# as the name implies we want to be able to test (paper trading) the strategy on a given day (today, but that info comes from the user)
# we don't actually connect to an exchange, just run the strategy and return our results

from src.inference.strategy import simulate_inference
from src.inference.inference_utils import validate_ticker, validate_date
from src.inference.chart import build_candlestick_data

UNIVERSE = ["AAPL", "AMZN", "AVGO", "GOOG", "GOOGL", "META", "MSFT", "NVDA", "TSLA", "WMT"]


async def run_daily_inference(ticker: str, date: str) -> dict:
    """Run a single-day paper-trade inference for a given ticker.

    Validates inputs, runs the strategy, builds a candlestick chart
    from dataset start to the specified date, and returns both in a
    single response dict.

    Args:
        ticker: Stock symbol from the user.
        date: Date string from the user (YYYY-MM-DD).

    Returns:
        Dict with inference results, chart_html, or error info.
    """
    try:
        if not validate_ticker(ticker, UNIVERSE):
            return {"error": f"Ticker '{ticker}' not in universe: {UNIVERSE}"}

        if not validate_date(date):
            return {"error": f"Invalid date format: '{date}'. Expected YYYY-MM-DD."}

        result = simulate_inference(ticker=ticker, date=date)
        result["chart_data"] = build_candlestick_data(ticker=ticker, date=date)
        return result

    except Exception as e:
        return {"error": f"Daily inference failed: {e}"}


async def run_batch_inference(date: str) -> dict:
    """Run inference for all tickers in the universe on a given date.

    Loops the full universe, collects predictions and chart data,
    and computes summary statistics across all tickers.

    Args:
        date: Date string (YYYY-MM-DD).

    Returns:
        Dict with 'results' (list of per-ticker dicts) and 'summary' stats.
    """
    try:
        if not validate_date(date):
            return {"error": f"Invalid date format: '{date}'. Expected YYYY-MM-DD."}

        results = []
        for ticker in UNIVERSE:
            result = simulate_inference(ticker=ticker, date=date)
            if "error" not in result:
                result["chart_data"] = build_candlestick_data(ticker=ticker, date=date)
            results.append(result)

        # Filter successful predictions for summary stats
        valid = [r for r in results if "error" not in r]

        if not valid:
            return {"error": "No valid predictions for any ticker on this date."}

        # Summary statistics
        top = max(valid, key=lambda r: r.get("model_confidence", 0))
        avg_conf = sum(r.get("model_confidence", 0) for r in valid) / len(valid)
        correct_count = sum(1 for r in valid if r.get("model_correct"))

        summary = {
            "date": date,
            "n_tickers": len(valid),
            "n_errors": len(results) - len(valid),
            "top_ticker": top.get("ticker", "?"),
            "top_prediction": top.get("model_bucket", "?"),
            "top_confidence": round(top.get("model_confidence", 0), 4),
            "avg_confidence": round(avg_conf, 4),
            "accuracy": round(correct_count / len(valid), 4) if valid else 0,
            "sample_type": valid[0].get("sample_type", "?"),
            "model": "LGBM 3-Class Moneyness",
        }

        return {"results": results, "summary": summary}

    except Exception as e:
        return {"error": f"Batch inference failed: {e}"}
