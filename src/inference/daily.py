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
