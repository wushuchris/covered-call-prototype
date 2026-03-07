import pandas as pd

BASE = "https://validex-ml-data.s3.us-east-1.amazonaws.com"

daily    = pd.read_parquet(f"{BASE}/daily_adjusted/ALL_daily_adjusted.parquet")
income   = pd.read_parquet(f"{BASE}/fundamentals/ALL_income_statement.parquet")
balance  = pd.read_parquet(f"{BASE}/fundamentals/ALL_balance_sheet.parquet")
cashflow = pd.read_parquet(f"{BASE}/fundamentals/ALL_cash_flow.parquet")
options  = pd.read_parquet(f"{BASE}/options/ALL_options.parquet")
overview = pd.read_csv(f"{BASE}/fundamentals/ALL_overview.csv")

print(daily.shape, income.shape, options.shape)