# inference utils are going to be very important.
# we not only want to verify data integrity, but also compute additional features if needed or data drift metrics
# given we are using placeholder inference strategies for now, it might contain just a limited amount of stuff

from src.utils import ServiceRequest


def unpack_request(req: ServiceRequest) -> ServiceRequest:
    """Validate and return the incoming ServiceRequest.

    In production this will also handle data integrity checks
    and feature computation. For now it passes through.

    Args:
        req: ServiceRequest already parsed by FastAPI.

    Returns:
        The same ServiceRequest instance.
    """
    try:
        return req
    except Exception as e:
        raise ValueError(f"Failed to unpack request: {e}")


def validate_ticker(ticker: str, universe: list[str]) -> bool:
    """Check that the requested ticker is in the allowed universe.

    Args:
        ticker: Symbol to validate.
        universe: List of allowed symbols.

    Returns:
        True if valid, False otherwise.
    """
    return ticker.upper() in [t.upper() for t in universe]


def validate_date(date_str: str) -> bool:
    """Basic date format validation (YYYY-MM-DD).

    Args:
        date_str: Date string to validate.

    Returns:
        True if parseable, False otherwise.
    """
    from datetime import datetime
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
        return True
    except (ValueError, TypeError):
        return False
