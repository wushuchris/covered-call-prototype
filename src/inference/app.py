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
from src.inference.daily import run_daily_inference
from src.inference.backtesting import run_backtest

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


@app.post("/backtest")
async def backtest_endpoint(body: ServiceRequest):
    """Receive a backtest request and run the backtesting pipeline.

    Args:
        body: ServiceRequest parsed from JSON by FastAPI.

    Returns:
        Dict with backtest report.
    """
    try:
        result = await run_backtest()
        return result
    except Exception as e:
        logger.error(f"Error on /backtest: {e}")
        return {"error": str(e)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8009)
