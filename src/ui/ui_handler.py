# for now placeholder functions, working on the ui rendering first

# the goal is to have this fasthtml service in charge of ui calls, and there is a twin FASTAPI microservice that handles the actual heavy-loads
# model backtesting and daily inference. the job of of the handler is to send info to the FASTAPI service (input)
# and receive the data output, process it, so it is then used by the @rt call to render the HTML component as needed (and return it to the client)

# the FASTAPI server (being developed simultaneously) is going to be hosted on localhost as well, port 8009. we want to use asynchronous requests all the way down
# (unless we do actually need a synchronous request, and it should be justified). inter-service communication should be by means of aiohttp
# (async, easy to understand). we want to standardize data handling, so instead of back and forthing multiple loose components a request_byte object
# (dataclass/basemodel) should be used. the @rt call does not need to know we are using said object. it just sends the input as a dictionary
# and the handler's first order of business is to turn it into the data object (which is to be shared between both services to allow for robust handling of data)
# the data object should be generic then (data, additional params; kind of like a header body format), and both services are in charge of unpacking and using
# as needed. we DO WANT TO keep it human friendly in terms of complexity. Simplicity is key.

# there exists a system wide utils function and a folder specific utils function, in this case ui_utils.py they should be used for functions that are used more
# than on once and it allows the keep the ui_handler a bit less dense (e.g. the data handler unpacking/packing)

import aiohttp
from src.utils import ServiceRequest
from src.ui.ui_utils import pack_request, send_to_inference


async def handle_inference_call(ticker: str, date: str) -> dict:
    """Handle an inference request from the UI.

    Packs the user input into a ServiceRequest, sends it to the
    FastAPI inference service, and returns the response dict for
    the @rt to render.

    Args:
        ticker: Stock symbol selected by the user.
        date: Date string selected by the user.

    Returns:
        Dict with inference results or error info.
    """
    try:
        req = pack_request(ticker=ticker, date=date)
        async with aiohttp.ClientSession() as session:
            result = await send_to_inference(session, req, "/inference")
        return result
    except Exception as e:
        return {"error": f"Inference call failed: {e}"}


async def handle_batch_inference(date: str) -> dict:
    """Handle a batch inference request for all tickers.

    Packs the date into a ServiceRequest and sends it to the
    FastAPI inference service's /inference_batch endpoint.

    Args:
        date: Date string selected by the user.

    Returns:
        Dict with per-ticker results and summary, or error info.
    """
    try:
        req = pack_request(ticker="", date=date)
        async with aiohttp.ClientSession() as session:
            result = await send_to_inference(session, req, "/inference_batch")
        return result
    except Exception as e:
        return {"error": f"Batch inference call failed: {e}"}


async def handle_model_metrics(year: str = "all",
                               sample_type: str = "all") -> dict:
    """Handle a model metrics request from the UI.

    Calls the inference service's /model_metrics endpoint with filters.

    Args:
        year: Year filter.
        sample_type: 'all', 'train', or 'test'.

    Returns:
        Dict with model metrics or error info.
    """
    try:
        async with aiohttp.ClientSession() as session:
            url = f"http://localhost:8009/model_metrics?year={year}&sample_type={sample_type}"
            async with session.get(url) as resp:
                return await resp.json()
    except Exception as e:
        return {"error": f"Model metrics call failed: {e}"}


async def handle_backtest_call(year: str = "all",
                               budget: float = 100_000) -> dict:
    """Handle a backtesting request from the UI.

    Packs year and budget into a ServiceRequest and sends it
    to the FastAPI inference service's /backtest endpoint.
    Returns all 3 presets + baseline in one response.

    Args:
        year: Year filter ('all' or e.g. '2020').
        budget: Dollar budget per month.

    Returns:
        Dict with backtest results for all presets or error info.
    """
    try:
        req = pack_request(ticker="", date="", year=year, budget=budget)
        async with aiohttp.ClientSession() as session:
            result = await send_to_inference(session, req, "/backtest")
        return result
    except Exception as e:
        return {"error": f"Backtest call failed: {e}"}
