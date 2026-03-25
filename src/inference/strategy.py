# Covered call strategy — shared between daily inference (UI) and backtesting (NautilusTrader).
#
# predict_bucket() from model.py is the single source of truth for decisions.
# This module wraps it for two callers:
#   1. simulate_inference(ticker, date) — called by daily.py for the FastHTML UI
#   2. CoveredCallStrategy.on_bar(bar) — called by NautilusTrader during backtests
#
# The strategy itself does no feature computation or model loading —
# it asks model.py "what bucket for this ticker on this date?" and acts on the answer.

from nautilus_trader.config import StrategyConfig
from nautilus_trader.trading.strategy import Strategy
from nautilus_trader.model import Bar

from src.inference.model import predict_bucket, initialize

# 6-class buckets (moneyness x maturity) — matches the model output
BUCKETS_6 = [
    "ATM_SHORT", "ATM_LONG",
    "OTM5_SHORT", "OTM5_LONG",
    "OTM10_SHORT", "OTM10_LONG",
]

BASELINE_BUCKET = "OTM10_SHORT"


# ── Daily inference entry point (called by daily.py → FastHTML UI) ───────────

def simulate_inference(ticker: str, date: str) -> dict:
    """Run model inference for a given ticker and date.

    Delegates entirely to model.predict_bucket(). Returns the full
    result dict including model prediction, ground truth, and comparison.

    Args:
        ticker: Stock symbol.
        date: Date string (YYYY-MM-DD).

    Returns:
        Dict with inference results or error info.
    """
    try:
        return predict_bucket(ticker, date)
    except Exception as e:
        return {"error": f"Inference failed: {e}"}


def baseline_strategy() -> str:
    """Baseline comparison strategy: always sell short-dated 10% OTM calls.

    Returns:
        The baseline bucket label.
    """
    return BASELINE_BUCKET


# ── NautilusTrader strategy ──────────────────────────────────────────────────
# Shared between backtest and live — same class, different execution context.
# on_bar receives monthly decision-point bars, queries the model,
# and logs (or submits) covered call orders for the selected bucket.

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

    Queries model.predict_bucket() on each monthly bar to decide
    which covered call bucket to sell. Currently logs decisions;
    order submission will be wired when backtesting.py is ready.

    Do NOT call self.clock or self.log in __init__ — system
    hasn't initialized yet. Use on_start() for setup.
    """

    def __init__(self, config: CoveredCallStrategyConfig) -> None:
        super().__init__(config)
        self.instrument_id_str = config.instrument_id
        self.bar_type_str = config.bar_type

        # Decision log — accumulated during the run, used for reporting
        self.decisions = []

    def on_start(self) -> None:
        """Called when the strategy is started.

        Ensures the model is loaded and feature store is ready.
        """
        initialize()
        self.log.info("CoveredCallStrategy started — model loaded.")

    def on_bar(self, bar: Bar) -> None:
        """Called on each bar update (monthly decision point).

        Queries the model for the optimal bucket and logs the decision.
        Order submission is a TODO for when venue/instruments are wired.

        Args:
            bar: The received bar data.
        """
        try:
            # Extract ticker and date from the bar
            ticker = self.instrument_id_str.split(".")[0] if self.instrument_id_str else "UNKNOWN"
            date = str(bar.ts_event)[:10]  # ISO date from timestamp

            result = predict_bucket(ticker, date)

            if "error" in result:
                self.log.warning(f"No prediction for {ticker} {date}: {result['error']}")
                return

            bucket = result["model_bucket"]
            confidence = result["model_confidence"]
            sample_type = result["sample_type"]

            self.log.info(
                f"Decision: {ticker} {date} → {bucket} "
                f"(confidence={confidence:.2%}, {sample_type})"
            )

            # Accumulate for reporting
            self.decisions.append(result)

            # TODO: translate bucket into order parameters and submit
            # e.g. select strike/expiry from the options chain matching the bucket,
            # then self.submit_order(...)

            # ── Future metrics (require venue/position tracking) ─────────
            # These will be computed here once order submission is wired:
            #
            # Transaction costs:
            #   - bid-ask spread at entry (from options chain)
            #   - commission model (per-contract flat fee)
            #
            # Risk-adjusted return:
            #   - rolling Sharpe ratio (monthly returns / monthly std)
            #   - max drawdown (peak-to-trough on cumulative P&L)
            #
            # Delta-hedged P&L:
            #   - isolate vol premium from directional exposure
            #   - P&L = premium collected - delta * stock move - hedging cost
            #
            # IV analysis:
            #   - IV at entry vs realized vol over holding period
            #   - IV rank at entry (was premium rich or cheap?)
            #   - IV term structure slope (short vs long dated)

        except Exception as e:
            self.log.error(f"on_bar failed: {e}")

    def on_stop(self) -> None:
        """Called when the strategy is stopped.

        Cancel open orders, close positions, and report summary.
        """
        n = len(self.decisions)
        if n > 0:
            correct = sum(1 for d in self.decisions if d.get("model_correct"))
            top2 = sum(1 for d in self.decisions if d.get("model_top2_hit"))
            self.log.info(
                f"Run complete: {n} decisions, "
                f"accuracy={correct/n:.1%}, top-2={top2/n:.1%}"
            )
        self.log.info("CoveredCallStrategy stopped.")
