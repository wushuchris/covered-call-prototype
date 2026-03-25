"""
Mirror script — reproduces the full data pipeline from notebooks 02→04→05
as a standalone Python script. Run once to generate all processed parquets
needed for model inference.

Usage:
    python data_scripts/build_processed.py

Outputs (saved to data/processed/):
    daily_clean.parquet
    income_clean.parquet
    balance_clean.parquet
    cashflow_clean.parquet
    options_clean.parquet
    overview_clean.parquet
    features.parquet
    modeling_data.parquet
    class_weights.json
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.utils.class_weight import compute_class_weight

BASE = "https://validex-ml-data.s3.us-east-1.amazonaws.com"
PROCESSED = Path(__file__).resolve().parent.parent / "data" / "processed"
UNIVERSE = ["AAPL", "AMZN", "AVGO", "GOOG", "GOOGL", "META", "MSFT", "NVDA", "TSLA", "WMT"]


# ── 1. Download raw data from S3 ────────────────────────────────────────────

def download_raw():
    """Download raw datasets from the S3 mirror."""
    print("Downloading raw data from S3...")
    raw = {
        "daily": pd.read_parquet(f"{BASE}/daily_adjusted/ALL_daily_adjusted.parquet"),
        "income": pd.read_parquet(f"{BASE}/fundamentals/ALL_income_statement.parquet"),
        "balance": pd.read_parquet(f"{BASE}/fundamentals/ALL_balance_sheet.parquet"),
        "cashflow": pd.read_parquet(f"{BASE}/fundamentals/ALL_cash_flow.parquet"),
        "options": pd.read_parquet(f"{BASE}/options/ALL_options.parquet"),
        "overview": pd.read_csv(f"{BASE}/fundamentals/ALL_overview.csv"),
    }
    for name, df in raw.items():
        print(f"  {name}: {df.shape}")
    return raw


# ── 2. Clean (notebook 02) ──────────────────────────────────────────────────

def clean_daily(df):
    """Clean daily price data."""
    df = df.copy()
    df.columns = df.columns.str.lower().str.strip()
    df = df.rename(columns={"adj_close": "adjusted_close", "split_coeff": "split_coefficient"})
    df["date"] = pd.to_datetime(df["date"])
    numeric = ["open", "high", "low", "close", "adjusted_close", "volume", "dividend", "split_coefficient"]
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.drop_duplicates(subset=["symbol", "date"])
    return df.sort_values(["symbol", "date"]).reset_index(drop=True)


def clean_fundamentals(df, date_col="fiscalDateEnding"):
    """Clean a fundamentals table (income, balance, cashflow)."""
    df = df.copy()
    df.columns = df.columns.str.strip()
    if "Symbol" in df.columns:
        df = df.rename(columns={"Symbol": "symbol"})
    df[date_col] = pd.to_datetime(df[date_col])
    exclude = ["symbol", date_col, "reportedCurrency"]
    for col in [c for c in df.columns if c not in exclude]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.drop_duplicates(subset=["symbol", date_col])
    return df.sort_values(["symbol", date_col]).reset_index(drop=True)


def clean_options(df):
    """Clean options data."""
    df = df.copy()
    df.columns = df.columns.str.lower().str.strip()
    df["expiration"] = pd.to_datetime(df["expiration"])
    if "trade_date" in df.columns:
        df["trade_date"] = pd.to_datetime(df["trade_date"])
    if "call_put" in df.columns:
        df["call_put"] = df["call_put"].str.upper().str.strip()
    df = df.drop_duplicates(subset=["contractid"]) if "contractid" in df.columns else df.drop_duplicates()
    sort_cols = ["symbol", "trade_date", "expiration", "strike"] if "trade_date" in df.columns else ["symbol", "expiration", "strike"]
    return df.sort_values(sort_cols).reset_index(drop=True)


def clean_overview(df):
    """Clean overview data."""
    df = df.copy()
    df = df.rename(columns={"Symbol": "symbol"})
    keep = [c for c in ["symbol", "Name", "Sector", "Industry", "Exchange",
                         "SharesOutstanding", "DividendYield", "Beta"] if c in df.columns]
    df = df[keep]
    df.columns = df.columns.str.lower()
    for col in ["sharesoutstanding", "dividendyield", "beta"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def run_cleaning(raw):
    """Run all cleaning steps. Returns dict of cleaned DataFrames."""
    print("Cleaning data...")
    cleaned = {
        "daily": clean_daily(raw["daily"]),
        "income": clean_fundamentals(raw["income"]),
        "balance": clean_fundamentals(raw["balance"]),
        "cashflow": clean_fundamentals(raw["cashflow"]),
        "options": clean_options(raw["options"]),
        "overview": clean_overview(raw["overview"]),
    }
    for name, df in cleaned.items():
        print(f"  {name}: {df.shape}")
    return cleaned


# ── 3. Feature engineering (notebook 04) ─────────────────────────────────────

def compute_technical_features(daily_df):
    """Compute technical features from daily price data."""
    df = daily_df.copy().sort_values(["symbol", "date"])
    df["daily_return"] = df.groupby("symbol")["adjusted_close"].pct_change()

    # Volatility (annualized)
    for window in [10, 21, 63]:
        df[f"vol_{window}d"] = df.groupby("symbol")["daily_return"].transform(
            lambda x: x.rolling(window).std() * np.sqrt(252)
        )

    # Momentum
    for window in [5, 21, 63]:
        df[f"mom_{window}d"] = df.groupby("symbol")["adjusted_close"].pct_change(window)

    # SMA ratios
    for window in [21, 50, 200]:
        sma = df.groupby("symbol")["adjusted_close"].transform(lambda x: x.rolling(window).mean())
        df[f"sma{window}"] = sma
        df[f"price_to_sma{window}"] = df["adjusted_close"] / sma

    df["sma21_above_sma50"] = (df["sma21"] > df["sma50"]).astype(int)
    df["sma50_above_sma200"] = (df["sma50"] > df["sma200"]).astype(int)

    # Drawdowns
    for window in [63, 252]:
        rolling_max = df.groupby("symbol")["adjusted_close"].transform(
            lambda x: x.rolling(window, min_periods=1).max()
        )
        df[f"drawdown_{window}d"] = (df["adjusted_close"] - rolling_max) / rolling_max

    # Volume ratio
    vol_sma = df.groupby("symbol")["volume"].transform(lambda x: x.rolling(21).mean())
    df["volume_ratio"] = df["volume"] / vol_sma

    # High vol regime
    vol_median = df.groupby("symbol")["vol_21d"].transform("median")
    df["high_vol_regime"] = (df["vol_21d"] > vol_median).astype(int)

    return df


def compute_fundamental_ratios(fund_df):
    """Compute derived fundamental ratios per ticker."""
    df = fund_df.copy().sort_values(["symbol", "fiscalDateEnding"])
    df["gross_margin"] = df["grossProfit"] / df["totalRevenue"].replace(0, np.nan)
    df["operating_margin"] = df["operatingIncome"] / df["totalRevenue"].replace(0, np.nan)
    df["net_margin"] = df["netIncome"] / df["totalRevenue"].replace(0, np.nan)
    df["revenue_growth_yoy"] = df.groupby("symbol")["totalRevenue"].pct_change(4)
    df["earnings_growth_yoy"] = df.groupby("symbol")["netIncome"].pct_change(4)
    df["debt_to_equity"] = df["shortLongTermDebtTotal"] / df["totalShareholderEquity"].replace(0, np.nan)
    df["cash_ratio"] = df["cashAndCashEquivalentsAtCarryingValue"] / df["totalLiabilities"].replace(0, np.nan)
    df["roe"] = df["netIncome"] / df["totalShareholderEquity"].replace(0, np.nan)
    df["roa"] = df["netIncome"] / df["totalAssets"].replace(0, np.nan)
    df["freeCashFlow"] = df.get("operatingCashflow", 0) - df.get("capitalExpenditures", 0).abs()
    return df


def build_features(cleaned):
    """Build the monthly feature dataset from cleaned data."""
    print("Engineering features...")
    daily = cleaned["daily"][cleaned["daily"]["symbol"].isin(UNIVERSE)].copy()
    income = cleaned["income"][cleaned["income"]["symbol"].isin(UNIVERSE)].copy()
    balance = cleaned["balance"][cleaned["balance"]["symbol"].isin(UNIVERSE)].copy()
    cashflow = cleaned["cashflow"][cleaned["cashflow"]["symbol"].isin(UNIVERSE)].copy()
    overview = cleaned["overview"]

    # Technical features
    daily_feat = compute_technical_features(daily)

    # Monthly decision points (last trading day of each month)
    daily_feat["year_month"] = daily_feat["date"].dt.to_period("M")
    monthly = daily_feat.groupby(["symbol", "year_month"]).last().reset_index()
    monthly["decision_date"] = monthly["date"]

    # Technical columns to keep
    tech_cols = [
        "vol_10d", "vol_21d", "vol_63d", "mom_5d", "mom_21d", "mom_63d",
        "price_to_sma21", "price_to_sma50", "price_to_sma200",
        "sma21_above_sma50", "sma50_above_sma200",
        "drawdown_63d", "drawdown_252d", "volume_ratio", "high_vol_regime",
    ]
    id_cols = ["symbol", "decision_date", "year_month", "adjusted_close", "volume"]
    monthly = monthly[id_cols + tech_cols].copy()

    # Fundamentals
    income_cols = [c for c in ["symbol", "fiscalDateEnding", "totalRevenue", "grossProfit",
                                "operatingIncome", "netIncome", "ebitda"] if c in income.columns]
    balance_cols = [c for c in ["symbol", "fiscalDateEnding", "totalAssets", "totalLiabilities",
                                 "totalShareholderEquity", "cashAndCashEquivalentsAtCarryingValue",
                                 "shortLongTermDebtTotal"] if c in balance.columns]
    cashflow_cols = [c for c in ["symbol", "fiscalDateEnding", "operatingCashflow",
                                  "capitalExpenditures"] if c in cashflow.columns]

    fund = income[income_cols].merge(
        balance[balance_cols], on=["symbol", "fiscalDateEnding"], how="outer"
    ).merge(
        cashflow[cashflow_cols], on=["symbol", "fiscalDateEnding"], how="outer"
    )
    for col in [c for c in fund.columns if c not in ["symbol", "fiscalDateEnding"]]:
        fund[col] = pd.to_numeric(fund[col], errors="coerce")
    fund = fund.sort_values(["symbol", "fiscalDateEnding"]).reset_index(drop=True)

    fund = compute_fundamental_ratios(fund)

    # Forward-fill fundamentals to monthly decision points (merge_asof)
    ratio_cols = [
        "gross_margin", "operating_margin", "net_margin",
        "revenue_growth_yoy", "earnings_growth_yoy",
        "debt_to_equity", "cash_ratio", "roe", "roa",
    ]
    result_parts = []
    for symbol in UNIVERSE:
        m = monthly[monthly["symbol"] == symbol].sort_values("decision_date")
        f = fund[fund["symbol"] == symbol].sort_values("fiscalDateEnding")
        if len(f) == 0:
            result_parts.append(m)
            continue
        merged = pd.merge_asof(
            m, f[["fiscalDateEnding"] + ratio_cols],
            left_on="decision_date", right_on="fiscalDateEnding",
            direction="backward",
        )
        result_parts.append(merged)
    monthly = pd.concat(result_parts, ignore_index=True)

    # Valuation features
    shares = overview[["symbol", "sharesoutstanding"]].copy()
    shares["sharesoutstanding"] = pd.to_numeric(shares["sharesoutstanding"], errors="coerce")
    monthly = monthly.merge(shares, on="symbol", how="left")

    monthly["market_cap"] = monthly["adjusted_close"] * monthly["sharesoutstanding"]
    monthly["revenue_ttm"] = monthly.get("totalRevenue", pd.Series(dtype=float)) * 4
    monthly["earnings_ttm"] = monthly.get("netIncome", pd.Series(dtype=float)) * 4

    # Use fundamentals already merged for valuation
    fund_for_val = fund[["symbol", "fiscalDateEnding", "totalRevenue", "netIncome", "ebitda",
                          "operatingCashflow", "capitalExpenditures",
                          "shortLongTermDebtTotal", "cashAndCashEquivalentsAtCarryingValue"]].copy()
    val_parts = []
    for symbol in UNIVERSE:
        m = monthly[monthly["symbol"] == symbol].sort_values("decision_date")
        f = fund_for_val[fund_for_val["symbol"] == symbol].sort_values("fiscalDateEnding")
        if len(f) == 0:
            val_parts.append(m)
            continue
        merged = pd.merge_asof(
            m, f.drop(columns=["symbol"]),
            left_on="decision_date", right_on="fiscalDateEnding",
            direction="backward", suffixes=("", "_fund"),
        )
        val_parts.append(merged)
    monthly = pd.concat(val_parts, ignore_index=True)

    # Compute valuation ratios
    rev_ttm = (monthly["totalRevenue"] * 4).replace(0, np.nan)
    earn_ttm = (monthly["netIncome"] * 4).replace(0, np.nan)
    ebitda_ttm = (monthly["ebitda"] * 4).replace(0, np.nan) if "ebitda" in monthly.columns else pd.Series(np.nan, index=monthly.index)
    fcf_ttm = ((monthly.get("operatingCashflow", 0) - monthly.get("capitalExpenditures", 0).abs()) * 4).replace(0, np.nan)
    ev = monthly["market_cap"] + monthly.get("shortLongTermDebtTotal", 0).fillna(0) - monthly.get("cashAndCashEquivalentsAtCarryingValue", 0).fillna(0)

    monthly["pe_ratio"] = monthly["market_cap"] / earn_ttm
    monthly["ps_ratio"] = monthly["market_cap"] / rev_ttm
    monthly["ev_ebitda"] = ev / ebitda_ttm
    monthly["fcf_yield"] = fcf_ttm / monthly["market_cap"].replace(0, np.nan)

    # Final feature selection
    feature_cols = tech_cols + ratio_cols + ["pe_ratio", "ps_ratio", "ev_ebitda", "fcf_yield"]
    final_cols = ["symbol", "decision_date", "year_month", "adjusted_close", "volume"] + feature_cols
    existing = [c for c in final_cols if c in monthly.columns]
    features = monthly[existing].copy()

    # Drop early rows without enough history
    features = features[features["decision_date"] >= "2001-01-01"].copy()
    features = features.sort_values(["symbol", "decision_date"]).reset_index(drop=True)

    print(f"  Features: {features.shape} ({len(feature_cols)} features)")
    return features


# ── 4. Label construction (notebook 05) ──────────────────────────────────────

def build_labels(features, cleaned):
    """Construct labels: best-performing bucket per (ticker, month)."""
    print("Constructing labels...")
    options = cleaned["options"][cleaned["options"]["symbol"].isin(UNIVERSE)].copy()
    daily = cleaned["daily"][cleaned["daily"]["symbol"].isin(UNIVERSE)].copy()

    calls = options[options["call_put"] == "CALL"].copy()
    calls["expiration"] = pd.to_datetime(calls["expiration"])
    calls["trade_date"] = pd.to_datetime(calls["trade_date"])
    calls["dte"] = (calls["expiration"] - calls["trade_date"]).dt.days
    daily["date"] = pd.to_datetime(daily["date"])

    # Assign buckets
    def assign_moneyness(delta):
        if 0.15 <= delta < 0.30:
            return "OTM10"
        elif 0.30 <= delta < 0.45:
            return "OTM5"
        elif 0.45 <= delta <= 0.60:
            return "ATM"
        return None

    def assign_maturity(dte):
        if 7 <= dte <= 37:
            return "DTE30"
        elif 38 <= dte <= 75:
            return "DTE60"
        elif 76 <= dte <= 120:
            return "DTE90"
        return None

    calls["moneyness"] = calls["delta"].apply(assign_moneyness)
    calls["maturity"] = calls["dte"].apply(assign_maturity)
    calls = calls.dropna(subset=["moneyness", "maturity"])
    calls["bucket"] = calls["moneyness"] + "_" + calls["maturity"]

    # Merge entry price
    daily_prices = daily[["symbol", "date", "adjusted_close"]].rename(
        columns={"date": "trade_date", "adjusted_close": "entry_price"}
    )
    calls = calls.merge(daily_prices, on=["symbol", "trade_date"], how="left")

    # Merge exit price (merge_asof per symbol)
    daily_exit = daily[["symbol", "date", "adjusted_close"]].rename(
        columns={"date": "expiration", "adjusted_close": "exit_price"}
    )
    parts = []
    for sym in UNIVERSE:
        left = calls[calls["symbol"] == sym].sort_values("expiration")
        right = daily_exit[daily_exit["symbol"] == sym].sort_values("expiration")
        if len(left) == 0 or len(right) == 0:
            continue
        merged = pd.merge_asof(left, right[["expiration", "exit_price"]], on="expiration", direction="backward")
        parts.append(merged)
    calls = pd.concat(parts, ignore_index=True)

    # Compute payoff
    calls["premium"] = calls["mark"].fillna((calls["bid"] + calls["ask"]) / 2)
    calls["stock_pnl"] = np.minimum(calls["exit_price"], calls["strike"]) - calls["entry_price"]
    calls["return"] = (calls["premium"] + calls["stock_pnl"]) / calls["entry_price"]

    # Filter outliers
    calls = calls[(calls["return"] > -0.5) & (calls["return"] < 0.5)]
    calls = calls[calls["entry_price"] > 0]
    calls = calls[calls["premium"] > 0]

    # Aggregate to monthly bucket returns
    calls["year_month"] = calls["trade_date"].dt.to_period("M")
    bucket_returns = calls.groupby(["symbol", "year_month", "bucket"]).agg(
        mean_return=("return", "mean"),
    ).reset_index()

    # Best bucket per (ticker, month)
    best_idx = bucket_returns.groupby(["symbol", "year_month"])["mean_return"].idxmax()
    best = bucket_returns.loc[best_idx][["symbol", "year_month", "bucket", "mean_return"]].copy()
    best = best.rename(columns={"bucket": "best_bucket", "mean_return": "best_return"})

    # Label encoding
    bucket_to_label = {
        "ATM_DTE30": 0, "ATM_DTE60": 1, "ATM_DTE90": 2,
        "OTM5_DTE30": 3, "OTM5_DTE60": 4, "OTM5_DTE90": 5,
        "OTM10_DTE30": 6, "OTM10_DTE60": 7, "OTM10_DTE90": 8,
    }
    best["label"] = best["best_bucket"].map(bucket_to_label)

    # Merge with features
    features["year_month"] = features["decision_date"].dt.to_period("M")
    modeling_data = features.merge(
        best[["symbol", "year_month", "best_bucket", "best_return", "label"]],
        on=["symbol", "year_month"],
        how="inner",
    )
    modeling_data = modeling_data.dropna(subset=["label"])
    modeling_data["label"] = modeling_data["label"].astype(int)

    # Class weights
    classes = np.array(sorted(modeling_data["label"].unique()))
    weights = compute_class_weight("balanced", classes=classes, y=modeling_data["label"])
    class_weights = {int(c): float(w) for c, w in zip(classes, weights)}

    print(f"  Modeling data: {modeling_data.shape}")
    print(f"  Classes: {len(class_weights)}")
    return modeling_data, class_weights


# ── 5. Main ──────────────────────────────────────────────────────────────────

def main():
    """Run the full pipeline: download → clean → features → labels → save."""
    PROCESSED.mkdir(parents=True, exist_ok=True)

    # Download
    raw = download_raw()

    # Clean
    cleaned = run_cleaning(raw)
    del raw  # free memory

    # Save cleaned
    cleaned["daily"].to_parquet(PROCESSED / "daily_clean.parquet", index=False)
    cleaned["income"].to_parquet(PROCESSED / "income_clean.parquet", index=False)
    cleaned["balance"].to_parquet(PROCESSED / "balance_clean.parquet", index=False)
    cleaned["cashflow"].to_parquet(PROCESSED / "cashflow_clean.parquet", index=False)
    cleaned["options"].to_parquet(PROCESSED / "options_clean.parquet", index=False)
    cleaned["overview"].to_parquet(PROCESSED / "overview_clean.parquet", index=False)
    print("Cleaned data saved.")

    # Features
    features = build_features(cleaned)
    features.to_parquet(PROCESSED / "features.parquet", index=False)

    # Labels
    modeling_data, class_weights = build_labels(features, cleaned)
    modeling_data.to_parquet(PROCESSED / "modeling_data.parquet", index=False)
    with open(PROCESSED / "class_weights.json", "w") as f:
        json.dump(class_weights, f, indent=2)

    print(f"\nAll outputs saved to {PROCESSED}/")
    print("Done.")


if __name__ == "__main__":
    main()
