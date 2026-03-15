import logging
import os
import glob
import time
import functools
from datetime import datetime, timedelta
from pydantic import BaseModel, Field


# shared data structure for inter-service communication (header-body format)
# both services pack/unpack using this same structure. generic enough that
# the @rt call just sends a dictionary and the handler turns it into this object.
# kept human-friendly — data carries the payload, params holds metadata.
class ServiceRequest(BaseModel):
    """Shared request model for inter-service communication.

    Works as both a Pydantic model (FastAPI body parsing) and
    a simple serializable structure (aiohttp JSON transport).
    """
    data: dict = Field(default_factory=dict)
    params: dict = Field(default_factory=dict)


# for now the key main functions are going to be custom logger creation (we want to define a function that generates a new logger)
# that way the microservices can call this function and spawn per project loggers
# we want the logger to print to console errors only, log everything to .txt files (from INFO to ERROR), and delete logs more than a day old (we don't want to
# accumulate .txt files, just present day stuff)
# the log should be brief, date, description, file where it broke
# it is worth mentioning that INFO is mainly going to be handled by the decorator below, and errors will be handled by try catch (Exception)
# blocks, where the caught exception will simply do something like custom_logger.error(f"Error {e} on xyz. Please..."")
def create_logger(name: str, log_dir: str = "logs") -> logging.Logger:
    """Create a per-service logger.

    Console: ERROR only.
    File: INFO and above, written to `logs/<name>_<date>.txt`.
    Stale logs (>1 day old) are deleted on creation.

    Args:
        name: Logger/service name (e.g. 'ui', 'inference').
        log_dir: Directory for log files.

    Returns:
        Configured logging.Logger instance.
    """
    try:
        os.makedirs(log_dir, exist_ok=True)
        _cleanup_old_logs(log_dir)

        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)

        # avoid duplicate handlers on repeated calls
        if logger.handlers:
            return logger

        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                                datefmt="%Y-%m-%d %H:%M:%S")

        # console — errors only
        ch = logging.StreamHandler()
        ch.setLevel(logging.ERROR)
        ch.setFormatter(fmt)
        logger.addHandler(ch)

        # file — INFO and above, one file per day
        today = datetime.now().strftime("%Y-%m-%d")
        fh = logging.FileHandler(os.path.join(log_dir, f"{name}_{today}.txt"))
        fh.setLevel(logging.INFO)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

        return logger

    except Exception as e:
        # fallback: return a basic logger so the caller never crashes
        fallback = logging.getLogger(name)
        fallback.warning(f"Logger setup failed: {e}")
        return fallback


def _cleanup_old_logs(log_dir: str) -> None:
    """Delete log files older than 1 day."""
    cutoff = time.time() - timedelta(days=1).total_seconds()
    for path in glob.glob(os.path.join(log_dir, "*.txt")):
        try:
            if os.path.getmtime(path) < cutoff:
                os.remove(path)
        except OSError:
            pass


# we want a tool/function decorator that can be placed on  regular functions and acts as a wrapper that loggs the INFO
# date, time to completion, function name, file
# function takes care of only logging INFO for regular functions. handling generators, routes, functions where another decorator is present is handled differently
# basic log for basic functions
def log_call(logger: logging.Logger):
    """Decorator that logs function name, file, and execution time at INFO level.

    Usage:
        logger = create_logger("ui")

        @log_call(logger)
        def my_function():
            ...

    Args:
        logger: Logger instance to write to.

    Returns:
        Decorator function.
    """
    def decorator(fn):
        @functools.wraps(fn)
        async def async_wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                result = await fn(*args, **kwargs)
                elapsed = time.perf_counter() - start
                logger.info(f"{fn.__name__} | {fn.__code__.co_filename} | {elapsed:.4f}s")
                return result
            except Exception as e:
                logger.error(f"{fn.__name__} | {fn.__code__.co_filename} | {e}")
                raise

        @functools.wraps(fn)
        def sync_wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                result = fn(*args, **kwargs)
                elapsed = time.perf_counter() - start
                logger.info(f"{fn.__name__} | {fn.__code__.co_filename} | {elapsed:.4f}s")
                return result
            except Exception as e:
                logger.error(f"{fn.__name__} | {fn.__code__.co_filename} | {e}")
                raise

        import asyncio
        if asyncio.iscoroutinefunction(fn):
            return async_wrapper
        return sync_wrapper
    return decorator
