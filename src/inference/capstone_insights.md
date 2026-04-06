# Capstone Report Insights — Claude Analysis Context
# Extracted from AAI-590-G6-Capstone-Report.pdf (20 pages)
# This file is loaded as context for the Claude analysis node (Graph 4).

## Project Framing

- Multi-class classification: predict optimal covered call strategy bucket (moneyness x maturity)
- 10 large-cap US stocks, monthly decision frequency, 2015-2025 data
- 9 original classes (ATM/OTM5/OTM10 x 30/60/90 DTE), consolidated to 7 for deep learning due to sparsity in long-dated categories
- LGBM production model uses 3 classes (moneyness only: ATM, OTM5, OTM10) with maturity assigned post-prediction via IV rank

## Model Performance — Key Numbers

### LGBM 3-Class (Production)
- Walk-forward validation macro F1: 0.47
- 2025 test set: 63% accuracy, 0.59 macro F1
- 34 features (15 technical, 9 fundamental, 4 valuation, 6 IV-derived)
- Feature selection via Random Forest importance on training set only

### LSTM-CNN 7-Class (Dashboard)
- Best variant (regularised): 0.110 macro F1, 38.1% accuracy on test
- Training F1: 0.819, validation F1: 0.206 — severe overfitting gap of 0.613
- Overfitting began diverging early, validation loss increased sharply after epoch 7

### XGBoost 7-Class (Baseline)
- Training F1: 0.830, test F1: 0.142 — poor generalization
- Achieved F1 of 0.48 for the dominant class only

### PatchTST Transformer
- Underperformed simpler tree-based methods
- Data-hungry architecture not appropriate for moderately sized financial datasets

### Random Baseline
- 7-class random accuracy: 14.3%
- Best model (34% accuracy) represents 2.4x improvement over random

## Critical Limitations

### 1. Distribution Shift (Most Important)
- Model performance is fundamentally constrained by distribution shift, not model architecture
- OTM10_60_90 class went from 1.25% of training data to 53.15% of 2024 test data
- This caused all models to heavily favor the dominant class, failing to generalize
- Performance varies >2.5x between years in walk-forward validation
- Models trained on historical data may fail when market conditions change

### 2. Overfitting
- ALL models exhibited significant overfitting (large train-test gaps)
- Increasing model complexity did NOT improve performance — the limitation is in the data
- LSTM-CNN Multi-Scale variant achieved comparable training performance but failed on test

### 3. Class Imbalance
- Significant imbalance across strategy buckets
- ATM_30 dominates (count ~12,000), OTM10_60 is sparse (~1,000)
- Partial resampling + class-weighted loss applied but insufficient
- 4 of 7 classes received F1 scores of zero across most models

### 4. Feature Limitations
- Macroeconomic features (FRED data) did NOT contribute meaningfully to predictive performance
- Stock-level technical and fundamental features proved more informative
- Highly correlated feature pairs: operating/net margin (0.95), 21d momentum/price-to-SMA50 (0.88), 10d/21d volatility (0.82)
- Only 16.2% of options contracts (52,184 calls) fell within relevant strike/maturity ranges

### 5. Practical Limitations
- System requires minimum 50-day feature history for predictions
- Lacks real-time data ingestion
- Outputs strategy bucket, not specific executable contract
- Additional post-processing needed for practical trading

## Model Strengths

### 1. Predictive Signal Exists
- All models exceeded random baseline, confirming features contain useful information
- The combination of technical, fundamental, and option-based features provides meaningful signal

### 2. Dominant Regime Identification
- Model performs well in identifying the dominant strategy bucket in regimes where OTM longer-dated options are optimal
- XGBoost achieved F1 of 0.48 for the dominant class

### 3. Feature Engineering Value
- IV features (mean, median, skew, term structure, rank) add significant predictive power
- Option Greeks (delta, gamma, theta, vega) are among strongest predictors per literature
- Technical indicators capture short/medium-term price dynamics relevant to option premiums

### 4. Regime-Aware Design
- Walk-forward validation preserves temporal integrity
- Strictly chronological splits prevent lookahead bias
- Train < 2022, Validation 2022-2023, Test 2024+

## Strategy Context

### Covered Call Trade-offs (from Literature)
- ATM options: greater premium income + downside protection
- OTM options: higher upside participation under certain market conditions
- Short-dated (30 DTE): favorable risk premia per empirical evidence
- Optimal strike selection depends on firm fundamentals, valuation context, and prevailing market conditions

### Data Quality Notes
- Missing data in financial statements: 27-56% missingness depending on variable
- Most missing values = fields not applicable to certain companies, not true data gaps
- Financial ratios clipped to reasonable ranges: P/E [-100, 500], FCF yield [-1, 1]
- Debt-to-equity ratio had 7 missing observations (1%), imputed with median
- Options data deduplicated from 1.85M to 927K unique contracts

## What This Means for the Analysis Node

When synthesizing predictions, Claude should:
1. Flag when predictions fall in historically sparse/shifting classes (especially OTM10_60_90)
2. Note that LGBM 3-class is significantly more reliable than LSTM-CNN 7-class
3. Weight confidence scores cautiously — high confidence doesn't mean high accuracy due to overfitting patterns
4. Consider the regime context: if current market conditions differ from training data, predictions are less reliable
5. Note that the model is a decision-support tool, not a trading signal — additional post-processing is always needed
6. Acknowledge that OTM10 baseline strategy has historically been hard to beat
7. When models disagree, the LGBM prediction should generally be trusted more (0.59 F1 vs 0.11 F1)
