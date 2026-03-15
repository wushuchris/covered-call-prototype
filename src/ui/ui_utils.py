# should be filled up as we go

from src.utils import ServiceRequest


def pack_request(ticker: str, date: str, **extra) -> ServiceRequest:
    """Convenience builder for the most common request shape.

    Args:
        ticker: Stock symbol (e.g. 'AAPL').
        date: Date string (e.g. '2026-03-15').
        **extra: Additional params forwarded to ServiceRequest.params.

    Returns:
        ServiceRequest ready for transport.
    """
    return ServiceRequest(
        data={"ticker": ticker, "date": date},
        params=extra,
    )


async def send_to_inference(session, payload: ServiceRequest, endpoint: str) -> dict:
    """Send an async POST to the FastAPI inference service.

    Args:
        session: aiohttp.ClientSession instance.
        payload: ServiceRequest to serialize and send.
        endpoint: Route path (e.g. '/inference' or '/backtest').

    Returns:
        JSON response as a dict, or error dict on failure.
    """
    url = f"http://localhost:8009{endpoint}"
    try:
        async with session.post(url, json=payload.model_dump()) as resp:
            return await resp.json()
    except Exception as e:
        return {"error": str(e)}
