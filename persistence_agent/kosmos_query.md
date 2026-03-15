# Kosmos Query

## Prompt (submit this)

We are building a machine learning system to predict covered call option returns for 10 US equities (2015-2025). The models are Random Forest, MLP (feedforward neural network), and a Transformer-based tabular model. The baseline strategy is always selling a 30 DTE, 10% OTM call.

Our proposed dataset has one row per listed call option contract per monthly snapshot (~900K rows). Each row includes contract features (moneyness, DTE, implied volatility, delta, gamma, theta, vega, bid-ask spread, open interest) and stock-level features (momentum, realized volatility, moving averages, fundamentals). We also want to derive IV surface features from the full options board per snapshot — IV term structure slope, skew, put-call OI ratios, volume concentration — to enrich the feature set. The label is the realized covered call return for that specific contract: `(S_T - S_0 + premium - max(0, S_T - K)) / S_0`. The original project plan discretizes the action space into 9 buckets (ATM / 5% OTM / 10% OTM × 30 / 60 / 90 DTE) and frames it as classification, but we want to also explore contract-level regression with post-inference ranking.

Search the literature for how covered call strategies, options return prediction, and volatility surface features have been modeled with ML. Explore what the regression target should be for the regression exploration — we have no prior on the right label transformation. Identify data quality filters, feature importance, and whether relationships are stable across bull/bear regimes.

## Upload files
- `ALL_options.parquet` — 1.85M rows, monthly options chain snapshots (calls + puts, all strikes/expirations)
- `ALL_daily_adjusted.parquet` — 27.5K rows, daily OHLCV + adj_close for 10 tickers

## Notes
- Job type: KOSMOS (full run — literature search + data analysis)
- Cost: 200 credits (of 650 free academic credits)
- Upload only options + daily prices. Fundamentals (income, balance, cashflow) available for follow-up.
