# this should define the backtesting strategy, we want a shared backtest-live srategy using Nautilus trader.
# we want to keep it simple at first, simulating model inference.
# i.e we don't care about how we came to the results (we do post-inferencing stuff only)
# for placeholder info we are therefore setting up an equally weighted (each bucket gets an x % percent chance of being chosen)
# strategy. we do this by using random choice function

import random

from nautilus_trader.config import StrategyConfig
from nautilus_trader.trading.strategy import Strategy
from nautilus_trader.model import Bar

BUCKETS = [
    "ATM_30DTE", "ATM_60DTE", "ATM_90DTE",
    "OTM5_30DTE", "OTM5_60DTE", "OTM5_90DTE",
    "OTM10_30DTE", "OTM10_60DTE", "OTM10_90DTE",
]


# ── Placeholder functions (used by daily inference and tests) ────────────────

def random_bucket_strategy() -> str:
    """Placeholder strategy: equally weighted random bucket selection.

    Each of the 9 buckets (ATM/OTM5/OTM10 x 30/60/90 DTE) has
    an equal probability of being chosen.

    Returns:
        String label of the selected bucket.
    """
    try:
        return random.choice(BUCKETS)
    except Exception as e:
        raise RuntimeError(f"Strategy selection failed: {e}")


def baseline_strategy() -> str:
    """Baseline comparison strategy: always sell 30 DTE, 10% OTM calls.

    Returns:
        The baseline bucket label.
    """
    return "OTM10_30DTE"


def simulate_inference(ticker: str, date: str) -> dict:
    """Simulate a model inference call with placeholder strategy.

    In production this will load the trained model and predict.
    For now it picks a random bucket and returns placeholder stats.

    Args:
        ticker: Stock symbol.
        date: Date string.

    Returns:
        Dict with simulated inference results.
    """
    try:
        selected = random_bucket_strategy()
        baseline = baseline_strategy()
        return {
            "ticker": ticker,
            "date": date,
            "prediction": selected,
            "baseline": baseline,
            "sharpe": round(random.uniform(-0.5, 2.5), 4),
            "expected_return": round(random.uniform(-0.03, 0.08), 4),
            "status": "placeholder",
        }
    except Exception as e:
        return {"error": f"Inference simulation failed: {e}"}


# ── NautilusTrader strategy scaffold ─────────────────────────────────────────
# shared between backtest and live — same class, different execution context.
# on_bar receives monthly decision-point bars, runs the model (or placeholder),
# and submits covered call orders for the selected bucket.

class CoveredCallStrategyConfig(StrategyConfig):
    """Configuration for the covered call strategy.

    Attributes:
        instrument_id: The instrument to trade (e.g. AAPL equity).
        bar_type: Bar type string for monthly decision bars.
    """
    instrument_id: str = ""
    bar_type: str = ""


class CoveredCallStrategy(Strategy):
    """NautilusTrader strategy for covered call bucket selection.

    Placeholder: uses random_bucket_strategy() for decisions.
    Production: will load trained ML model and predict optimal bucket.

    Do NOT call self.clock or self.log in __init__ — system
    hasn't initialized yet. Use on_start() for setup.
    """

    def __init__(self, config: CoveredCallStrategyConfig) -> None:
        super().__init__(config)
        self.instrument_id_str = config.instrument_id
        self.bar_type_str = config.bar_type

    def on_start(self) -> None:
        """Called when the strategy is started.

        Subscribe to bar data and initialize any indicators.
        """
        self.log.info("CoveredCallStrategy started (placeholder mode).")

    def on_bar(self, bar: Bar) -> None:
        """Called on each bar update (monthly decision point).

        Placeholder: logs a random bucket selection.
        Production: run model inference and submit orders.

        Args:
            bar: The received bar data.
        """
        try:
            selected = random_bucket_strategy()
            self.log.info(f"Decision: {selected} (placeholder)")
        except Exception as e:
            self.log.error(f"on_bar failed: {e}")

    def on_stop(self) -> None:
        """Called when the strategy is stopped.

        Cancel open orders, close positions.
        """
        self.log.info("CoveredCallStrategy stopped.")
