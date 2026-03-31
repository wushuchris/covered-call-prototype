#!/usr/bin/env python3
"""Standalone backtest runner — generates and caches backtest reports.

Run this script to precompute backtest results for all presets and years.
The inference service will use the cached results instead of recomputing.

Usage:
    python src/notebooks/run_backtest.py                   # all years, all presets
    python src/notebooks/run_backtest.py --year 2020       # single year
    python src/notebooks/run_backtest.py --force            # ignore cache, recompute

Outputs:
    cache/backtest_all_all.json      — full history, all presets
    cache/backtest_all_2020.json     — single year, all presets
    (etc.)

Teammates can run this independently to regenerate the cache with
different parameters or after retraining models.
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.inference.backtesting import run_backtest_all


async def main():
    parser = argparse.ArgumentParser(description="Run covered call backtests")
    parser.add_argument("--year", default="all", help="Year to backtest (default: all)")
    parser.add_argument("--budget", type=float, default=100_000, help="Budget per month (default: 100000)")
    parser.add_argument("--force", action="store_true", help="Ignore cache, recompute")
    args = parser.parse_args()

    print(f"Running backtest: year={args.year}, budget=${args.budget:,.0f}, force={args.force}")
    print()

    result = await run_backtest_all(budget=args.budget, year=args.year, force=args.force)

    if "error" in result:
        print(f"ERROR: {result['error']}")
        return

    print(f"Period: {result['date_range']['start']} to {result['date_range']['end']} ({result['n_months']} months)")
    print(f"Source: {result.get('source', 'unknown')}")
    print()

    # Print baseline
    bm = result["baseline"]["metrics"]
    print(f"{'':15s} {'Baseline':>12s} {'Conservative':>14s} {'Balanced':>12s} {'Aggressive':>12s}")
    print("-" * 67)

    metrics = [
        ("Ann. Return", "annualized_return", lambda v: f"{v:.1%}"),
        ("Sharpe", "sharpe_ratio", lambda v: f"{v:.2f}"),
        ("Max Drawdown", "max_drawdown", lambda v: f"{v:.1%}"),
        ("Hit Rate", "hit_rate", lambda v: f"{v:.1%}"),
        ("Avg P/L", "avg_p_l", lambda v: f"{v:.2f}"),
    ]

    for label, key, fmt in metrics:
        baseline_val = fmt(bm[key])
        cons_val = fmt(result["presets"]["conservative"]["metrics"][key])
        bal_val = fmt(result["presets"]["balanced"]["metrics"][key])
        agg_val = fmt(result["presets"]["aggressive"]["metrics"][key])
        print(f"{label:15s} {baseline_val:>12s} {cons_val:>14s} {bal_val:>12s} {agg_val:>12s}")

    print()
    print("Results cached. The inference service will use these automatically.")


if __name__ == "__main__":
    asyncio.run(main())
