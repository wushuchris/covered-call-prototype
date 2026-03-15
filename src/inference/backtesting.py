# in charge of backtesting by means of Nautilus trader
# should be a one-time only (batch) type oif thing, as it is computationally expensive
# functions should therefore check first if a cached (report of sorts), exists already (as is done by the nautilus trader)
# report should be a placeholder for now (we don't have the models, the strategy, or data; there exists data on aws but we are still deciding as to what we)
# keep and what we remove

# generate the placeholder and the actual functions, just make sure the functions are being triggered by the placeholder
# data ingestion for the backtesting pipeline the utils functions are going to be in charge of, here we want clean tractable function calls

import os
import json
from pathlib import Path

from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.backtest.config import BacktestEngineConfig
from nautilus_trader.model import TraderId

from src.inference.strategy import CoveredCallStrategy, CoveredCallStrategyConfig

# cache dir relative to project root, not CWD
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CACHE_DIR = _PROJECT_ROOT / "cache"
BACKTEST_CACHE_FILE = CACHE_DIR / "backtest_report.json"


def _load_cached_report() -> dict | None:
    """Check if a cached backtest report exists and return it.

    Returns:
        Cached report dict, or None if no cache found.
    """
    try:
        if BACKTEST_CACHE_FILE.exists():
            return json.loads(BACKTEST_CACHE_FILE.read_text())
    except Exception:
        pass
    return None


def _save_cached_report(report: dict) -> None:
    """Persist a backtest report to the cache file.

    Args:
        report: Report dict to save.
    """
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        BACKTEST_CACHE_FILE.write_text(json.dumps(report, indent=2))
    except Exception:
        pass


def _generate_placeholder_report() -> dict:
    """Generate a placeholder backtest report.

    This stands in until real models, strategies, and data
    are wired up through NautilusTrader.

    Returns:
        Dict with placeholder performance metrics.
    """
    return {
        "status": "placeholder",
        "strategy": "random_bucket (equally weighted)",
        "baseline": "OTM10_30DTE (always)",
        "metrics": {
            "total_return": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "num_trades": 0,
        },
        "note": "Placeholder report. Real backtesting pending model training and data pipeline.",
    }


# ── NautilusTrader backtest scaffold ─────────────────────────────────────────
# production flow:
#   1. load data (parquets from S3 / data_scripts)
#   2. wrangle into NautilusTrader format (TradeTickDataWrangler or BarDataWrangler)
#   3. configure BacktestEngine (venue, instrument, starting balances)
#   4. add CoveredCallStrategy
#   5. engine.run()
#   6. extract reports (account, fills, positions)
#   7. cache report
#
# NOTE: BacktestEngine uses global singleton state — cannot run multiple
# instances in the same process. run sequentially and dispose between runs.

def _build_engine() -> BacktestEngine:
    """Create and configure a BacktestEngine instance.

    Placeholder: returns an engine with no data or venue.
    Production: will add venue, instruments, and historical data.

    Returns:
        Configured BacktestEngine.
    """
    config = BacktestEngineConfig(trader_id=TraderId("BACKTESTER-001"))
    engine = BacktestEngine(config=config)

    # TODO: add venue (e.g. engine.add_venue(...))
    # TODO: add instruments
    # TODO: wrangle and add historical data (bars/ticks)
    # TODO: add CoveredCallStrategy with config

    return engine


async def run_backtest() -> dict:
    """Run the backtesting pipeline (or return cached results).

    Checks for a cached report first to avoid redundant computation.
    If no cache exists, generates a placeholder and caches it.
    In production, this will run the NautilusTrader BacktestEngine.

    Returns:
        Dict with backtest report.
    """
    try:
        # check cache first — backtesting is expensive
        cached = _load_cached_report()
        if cached is not None:
            cached["source"] = "cache"
            return cached

        # no cache — generate placeholder (will be NautilusTrader engine.run() in production)
        # production code:
        #   engine = _build_engine()
        #   engine.run()
        #   report = {
        #       "account": engine.trader.generate_account_report(venue),
        #       "fills": engine.trader.generate_order_fills_report(),
        #       "positions": engine.trader.generate_positions_report(),
        #   }
        #   engine.dispose()

        report = _generate_placeholder_report()
        _save_cached_report(report)
        report["source"] = "generated"
        return report

    except Exception as e:
        return {"error": f"Backtest failed: {e}"}
