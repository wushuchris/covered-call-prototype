# Ideological Grounding: ML-Driven Covered Call Optimization
**Author**: Carlos Ortiz
**Project**: AAI-590 Capstone — Validex Growth Investors
**Version**: 0.1 (2026-03-07)

---

## 1. Thesis

A covered call strategy's performance is determined by the choice of strike and expiration — not by exotic modeling. We hypothesize that conventional machine learning, applied to well-understood financial features, can systematically improve this choice over naive heuristics.

This project is not an exercise in model complexity. It is an exercise in **disciplined feature engineering, reproducible experimentation, and honest evaluation**.

---

## 2. Guiding Principles

### 2.1 Reproducibility Above All

Every result must be reproducible by a classmate with no prior context. This means:
- All pipelines run end-to-end from a single notebook or script
- No hidden state, no manual steps, no "run this cell twice"
- Environment is fully specified (`environment.yaml`, `requirements-dev.txt`)
- Random seeds are fixed. Data splits are deterministic and temporal (no future leakage)

### 2.2 Conventional Tools, Unconventional Rigor

We use standard, well-documented libraries — not because they are easy, but because they are auditable:

| Layer | Tool | Why |
|---|---|---|
| Data manipulation | pandas, numpy | Industry standard, transparent operations |
| Feature engineering | pandas + scipy | Explicit formulas, no black-box transforms |
| Baseline models | scikit-learn, XGBoost, LightGBM | Interpretable, well-studied, SHAP-compatible |
| Neural network | PyTorch (MLP) | Required by capstone; kept minimal and explainable |
| Explainability | SHAP, feature importances | Every model must justify its decisions |
| Hyperparameter search | Optuna | Reproducible, stateful, Bayesian |
| Visualization | matplotlib, seaborn, plotly | Publication-quality, notebook-native |

We avoid proprietary libraries, over-engineered abstractions, and anything that cannot be inspected line-by-line.

### 2.3 Simplicity as a Feature

Complexity is not intelligence. A 48-feature model that achieves 55% accuracy on a 9-class problem is more valuable than a 500-feature model at 57% — if the former can be explained to a portfolio manager in 10 minutes.

Design choices follow from this:
- **Action space**: 9 discrete buckets (3 moneyness x 3 DTE), not continuous strike/expiration
- **Decision frequency**: Monthly, not daily — matches how real covered call strategies operate
- **Feature set**: Derived from first principles (valuation, momentum, volatility, regime) — not mined from an arbitrarily large search space
- **Evaluation**: Risk-adjusted returns (Sharpe), not just accuracy — because a model that's "right" but picks low-premium buckets is useless

### 2.4 Respect the Financial Structure

Options are not generic tabular data. The modeling choices must respect the domain:
- **No future leakage**: Labels are constructed from forward-looking payoffs. The temporal split must be airtight.
- **Volatility is the central variable**: Realized vs implied vol spread is the core signal for covered call writing. Features and models should be sensitive to this.
- **Greeks are descriptive, not predictive**: Delta, gamma, theta describe the contract's current state — they are useful as features but are not causal drivers of future performance.
- **Regime matters**: A model trained in a bull market should be evaluated on how it degrades in a drawdown. Robustness > peak accuracy.

### 2.5 Honest Evaluation

The baseline strategy (always sell 30 DTE, 10% OTM) is not a straw man — it is a legitimate, commonly used approach. If the ML system cannot beat it on risk-adjusted returns out-of-sample, then it has failed, and we report that honestly.

Evaluation includes:
- Out-of-sample classification accuracy (by class, not just aggregate)
- Strategy backtest: cumulative returns, Sharpe ratio, max drawdown
- Ablation analysis: which feature groups (fundamentals vs. price vs. regime) drive performance
- Sensitivity analysis: how performance changes across market conditions

---

## 3. Architecture Philosophy

### 3.1 Pipeline Over Monolith

Every stage of the system is a separable, testable unit:

```
[Data Ingestion] → [Feature Engineering] → [Label Construction] → [Model Training] → [Evaluation]
```

Each stage:
- Takes explicit inputs (parquets, configs)
- Produces explicit outputs (parquets, model artifacts, metrics)
- Can be run independently
- Is documented in a companion notebook

### 3.2 Notebook as Proof

The primary deliverable is not a Python package — it is a set of notebooks that demonstrate the pipeline, show intermediate results, and make the reasoning transparent. Scripts exist to support the notebooks, not the other way around.

### 3.3 Configuration Over Code

Parameters that change (ticker universe, date ranges, bucket definitions, hyperparameters) live in config files or notebook headers — not buried in function bodies.

---

## 4. Feature Philosophy

Features are organized by economic intuition, not by data source:

| Category | Signal | Intuition |
|---|---|---|
| **Momentum** | 1m, 3m, 6m, 12m returns | Trending stocks behave differently under covered calls |
| **Volatility** | Realized vol (20/60/120d), Parkinson, IV | Core driver of option premium — vol regime determines strike selection |
| **Valuation** | P/E, P/S, FCF yield, margins | Fundamentally cheap stocks may have different payoff profiles |
| **Balance sheet** | Debt/equity, current ratio, ROE/ROA | Financial health affects drawdown risk |
| **Growth** | Revenue/earnings growth QoQ, YoY | Growth trajectory influences both price path and vol dynamics |
| **Regime** | Drawdown depth, MA crossovers, near-52w-high | Market state determines which moneyness/DTE bucket is optimal |
| **Options** | Moneyness, DTE, spread, IV | Contract-level descriptors for the classification target |

Each feature exists because there is a financial hypothesis for why it should matter. No feature mining.

---

## 5. What This Project Is Not

- **Not a trading system**: This is a prototype ML pipeline. It does not execute trades.
- **Not a pricing model**: We do not price options. We classify which bucket to sell.
- **Not a deep learning showcase**: The neural network component satisfies a capstone requirement. The intellectual contribution is in the pipeline, feature engineering, and evaluation framework.
- **Not overfit to history**: If the model only works on AAPL 2020-2023, it has failed.

---I 

## 6. Success Criteria

1. The pipeline runs end-to-end from raw data to backtest results in under 30 minutes
2. At least one ML model outperforms the baseline on out-of-sample Sharpe ratio
3. Feature importance analysis identifies economically interpretable drivers
4. All results are reproducible from a clean environment using provided notebooks
5. The team can present the system and its results clearly in a 10-minute video
