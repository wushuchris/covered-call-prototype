# FAST API server, locally hosted on port 8009. similar logic as the FAST HTML server
# we use this space to set up the routes (app.post, gets). and trigger the corresponding backtesting, daily function
# custom logger for this service as well
# activated by the if __main.py__ clause at the bottom, we are going to be running the dev server for now which is spawned by a Popen process by a mainn.py  trigger

# we should start with an inference call method and a backtesting method
# the request is using a specific data structure (shared between both services)
# the app function is in charge of captuirng it, but unpacking is the job of the handler (backtesting/daily)
# async all the way down once again.

from fastapi import FastAPI
import uvicorn
from src.utils import create_logger, ServiceRequest
from src.inference.inference_utils import unpack_request
from src.inference.daily import run_daily_inference, run_batch_inference
from src.inference.chart import build_candlestick_data
from src.inference.model import compute_model_metrics
from src.inference.backtesting import run_backtest_all

logger = create_logger("inference")

app = FastAPI(title="USD Trader Inference Service")


@app.post("/inference")
async def inference_endpoint(body: ServiceRequest):
    """Receive an inference request, unpack it, and run daily inference.

    The request body follows the shared ServiceRequest format
    (data + params). FastAPI validates and parses it automatically
    via the Pydantic model. Unpacking is delegated to inference_utils.

    Args:
        body: ServiceRequest parsed from JSON by FastAPI.

    Returns:
        Dict with inference results.
    """
    try:
        req = unpack_request(body)
        ticker = req.data.get("ticker", "")
        date = req.data.get("date", "")
        result = await run_daily_inference(ticker=ticker, date=date)
        return result
    except Exception as e:
        logger.error(f"Error on /inference: {e}")
        return {"error": str(e)}


@app.post("/inference_batch")
async def inference_batch_endpoint(body: ServiceRequest):
    """Receive a batch inference request for all tickers on a given date.

    Args:
        body: ServiceRequest with date in data dict.

    Returns:
        Dict with per-ticker results and summary statistics.
    """
    try:
        req = unpack_request(body)
        date = req.data.get("date", "")
        result = await run_batch_inference(date=date)
        return result
    except Exception as e:
        logger.error(f"Error on /inference_batch: {e}")
        return {"error": str(e)}


@app.get("/chart_data")
async def chart_data_endpoint(ticker: str = "", date: str = ""):
    """Return OHLC candlestick data for a ticker up to a given date.

    Lightweight endpoint — no model inference, just price data.

    Args:
        ticker: Stock symbol.
        date: End date (YYYY-MM-DD).

    Returns:
        Dict with chart_data list for ApexCharts.
    """
    try:
        data = build_candlestick_data(ticker=ticker, date=date)
        return {"ticker": ticker, "date": date, "chart_data": data}
    except Exception as e:
        logger.error(f"Error on /chart_data: {e}")
        return {"error": str(e), "chart_data": []}


@app.get("/model_metrics")
async def model_metrics_endpoint(year: str = "all", sample_type: str = "all"):
    """Compute model performance metrics from the feature store.

    Args:
        year: Year filter ('all' or e.g. '2020').
        sample_type: 'all', 'in-sample', or 'out-of-sample'.

    Returns:
        Dict with accuracy, F1, per-class breakdown, per-year breakdown.
    """
    try:
        return compute_model_metrics(year=year, sample_type=sample_type)
    except Exception as e:
        logger.error(f"Error on /model_metrics: {e}")
        return {"error": str(e)}


@app.post("/backtest")
async def backtest_endpoint(body: ServiceRequest):
    """Receive a backtest request and run the backtesting pipeline.

    Extracts preset and budget from the ServiceRequest params.
    Scoring, allocation, and per-bucket return lookup all happen here.

    Args:
        body: ServiceRequest parsed from JSON by FastAPI.

    Returns:
        Dict with backtest report (strategy + baseline metrics).
    """
    try:
        req = unpack_request(body)
        year = req.params.get("year", "all")
        budget = float(req.params.get("budget", 100_000))
        result = await run_backtest_all(budget=budget, year=year)
        return result
    except Exception as e:
        logger.error(f"Error on /backtest: {e}")
        return {"error": str(e)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8009)
