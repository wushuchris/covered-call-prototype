# ── NautilusTrader reference (REMOVED from production) ──────────────────────
#
# This file preserves the NautilusTrader CoveredCallStrategy class that was
# originally scaffolded in strategy.py. It was removed because:
#   - The backtest loop runs in plain Python (scoring.py + backtesting.py)
#   - NautilusTrader was imported but never instantiated or executed
#   - No venues, instruments, or order execution were wired
#
# If NautilusTrader is reintroduced post-capstone (e.g., for execution-level
# backtesting with fills, slippage, margin), this class can be restored.
#
# Dependencies needed:
#   pip install nautilus_trader
#
# ─────────────────────────────────────────────────────────────────────────────
#
# from nautilus_trader.config import StrategyConfig
# from nautilus_trader.trading.strategy import Strategy
# from nautilus_trader.model import Bar
# from nautilus_trader.backtest.engine import BacktestEngine
# from nautilus_trader.backtest.config import BacktestEngineConfig
# from nautilus_trader.model import TraderId
#
# from src.inference.model import predict_bucket, initialize
#
#
# class CoveredCallStrategyConfig(StrategyConfig):
#     instrument_id: str = ""
#     bar_type: str = ""
#
#
# class CoveredCallStrategy(Strategy):
#     """NautilusTrader strategy for covered call bucket selection.
#
#     Queries model.predict_bucket() on each monthly bar to decide
#     which covered call bucket to sell.
#     """
#
#     def __init__(self, config: CoveredCallStrategyConfig) -> None:
#         super().__init__(config)
#         self.instrument_id_str = config.instrument_id
#         self.bar_type_str = config.bar_type
#         self.decisions = []
#
#     def on_start(self) -> None:
#         initialize()
#         self.log.info("CoveredCallStrategy started.")
#
#     def on_bar(self, bar: Bar) -> None:
#         ticker = self.instrument_id_str.split(".")[0] if self.instrument_id_str else "UNKNOWN"
#         date = str(bar.ts_event)[:10]
#         result = predict_bucket(ticker, date)
#         if "error" not in result:
#             self.decisions.append(result)
#             self.log.info(f"{ticker} {date} -> {result['model_bucket']}")
#
#     def on_stop(self) -> None:
#         n = len(self.decisions)
#         if n > 0:
#             correct = sum(1 for d in self.decisions if d.get("model_correct"))
#             self.log.info(f"Run complete: {n} decisions, accuracy={correct/n:.1%}")
#
#
# def build_backtest_engine() -> BacktestEngine:
#     """Scaffold for NautilusTrader execution-level backtesting.
#
#     Requires:
#       - Venue definition (e.g., NASDAQ simulator)
#       - Instrument specs (equity + options contracts)
#       - Historical data loaded via add_data()
#       - CoveredCallStrategy added to engine
#     """
#     config = BacktestEngineConfig(trader_id=TraderId("BACKTESTER-001"))
#     engine = BacktestEngine(config=config)
#     # engine.add_venue(...)
#     # engine.add_instrument(...)
#     # engine.add_data(...)
#     # engine.add_strategy(CoveredCallStrategy(CoveredCallStrategyConfig(...)))
#     return engine
