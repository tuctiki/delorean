
import pandas as pd
import qlib
from qlib.data import D
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from delorean.conf import QLIB_PROVIDER_URI, QLIB_REGION, ETF_LIST
from delorean.alphas.factors import get_production_factors

def check_values():
    qlib.init(provider_uri=QLIB_PROVIDER_URI, region=QLIB_REGION)
    
    date = "2026-01-14" # A recent trading day
    expressions, names = get_production_factors()
    
    fields = list(expressions) + ["Ref($close, -1) / $close - 1"]
    names_with_label = list(names) + ["label"]
    
    print(f"Checking values for {date}...")
    df = D.features(ETF_LIST, fields, start_time=date, end_time=date)
    df.columns = names_with_label
    df = df.dropna()
    
    if df.empty:
        print("No data found.")
        return

    print("\n--- Sample Data (Head 10) ---")
    print(df.head(10))
    
    print("\n--- Correlation on this day ---")
    print(df.corr(method="spearman"))
    
    print("\n--- Factor Statistics ---")
    print(df.describe())

if __name__ == "__main__":
    check_values()
