import os
import requests
import pandas as pd

BASE = "https://validex-ml-data.s3.us-east-1.amazonaws.com"

FILES = {
    "daily_adjusted/ALL_daily_adjusted.parquet": "data/ALL_daily_adjusted.csv",
    "fundamentals/ALL_income_statement.parquet": "data/ALL_income_statement.csv",
    "fundamentals/ALL_balance_sheet.parquet": "data/ALL_balance_sheet.csv",
    "fundamentals/ALL_cash_flow.parquet": "data/ALL_cash_flow.csv",
    "options/ALL_options.parquet": "data/ALL_options.csv",
    "fundamentals/ALL_overview.csv": "data/ALL_overview.csv"
}

# Create data directory if it doesn't exist
os.makedirs("data", exist_ok=True)

# Download files
for s3_path, local_path in FILES.items():
    url = f"{BASE}/{s3_path}"
    
    if not os.path.exists(local_path):
        print(f"Downloading {s3_path} ...")
        r = requests.get(url)
        r.raise_for_status()

        with open(local_path, "wb") as f:
            f.write(r.content)

        print(f"Saved to {local_path}")
    else:
        print(f"{local_path} already exists, skipping download.")

# Load files locally
daily    = pd.read_parquet("data/ALL_daily_adjusted.csv")
income   = pd.read_parquet("data/ALL_income_statement.csv")
balance  = pd.read_parquet("data/ALL_balance_sheet.csv")
cashflow = pd.read_parquet("data/ALL_cash_flow.csv")
options  = pd.read_parquet("data/ALL_options.csv")
overview = pd.read_csv("data/ALL_overview.csv")

print(daily.shape, income.shape, options.shape)