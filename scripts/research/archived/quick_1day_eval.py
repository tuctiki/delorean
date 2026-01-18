
import qlib
import pandas as pd
import numpy as np
from qlib.data import D
import os
import sys

# Setup path
sys.path.insert(0, os.getcwd())

from delorean.conf import QLIB_PROVIDER_URI, QLIB_REGION, ETF_LIST

def evaluate_1day_ic():
    qlib.init(provider_uri=QLIB_PROVIDER_URI, region=QLIB_REGION)
    
    candidates = {
        "Vol_Skew_20": "-1 * Skew($close / Ref($close, 1) - 1, 20)",
        "Alpha_Gen_8": "-1 * (Sum(-1 * (Log($open + 1e-5)), 5) + Std($high, 5))",
        "Vol_Price_Div_Rev": "Mean(-1 * Corr($close / Ref($close, 1), $volume / Ref($volume, 1), 10), 5)",
        "Selection_Trend": "Log(Abs($close / Ref($close, 20) - 1) + 1.0001) * Power(($close / Mean($close, 60) - 1), 2)",
        "Smart_Flow_Rev": "-1 * ($close - $low) / ($high - $low + 0.001) * (Mean($volume, 5) / Mean($volume, 20))",
        "Gap_Fill_Rev": "-1 * ($close - $open) / (Abs($open - Ref($close, 1)) + 0.001)",
        "Structural_Breakout": "($close - Min($low, 60)) / (Max($high, 60) - Min($low, 60) + 1e-4)",
        "MOM120": "$close / Ref($close, 120) - 1",
    }
    
    label_expr = "Ref($close, -1) / $close - 1"
    
    start_date = "2025-01-01"
    end_date = "2026-01-17"
    
    exprs = list(candidates.values())
    names = list(candidates.keys())
    
    df = D.features(ETF_LIST, exprs + [label_expr], start_time=start_date, end_time=end_date)
    df.columns = names + ["label"]
    df = df.dropna()
    
    print(f"\n--- 1-Day IC Performance (2025-Recent) ---")
    results = []
    for name in names:
        ic = df[name].corr(df["label"], method="spearman")
        print(f"{name:<20}: IC = {ic:.4f}")
        results.append({"Name": name, "IC": ic})
        
    # Also check our current refined ones that failed the 2025 1-day check
    failure_check = {
        "Mom_Persistence": "Sum(If($close > Ref($close, 1), 1, 0), 10) / 10",
        "Acceleration": "($close / Ref($close, 5) - 1) - (Ref($close, 5) / Ref($close, 10) - 1)",
    }
    
    df_fail = D.features(ETF_LIST, list(failure_check.values()), start_time=start_date, end_time=end_date)
    df_fail.columns = list(failure_check.keys())
    # Merge with label
    df_merged = df_fail.join(df["label"], how="inner").dropna()
    
    print(f"\n--- Re-Check Failures ---")
    for name in failure_check:
        ic = df_merged[name].corr(df_merged["label"], method="spearman")
        print(f"{name:<20}: IC = {ic:.4f}")

if __name__ == "__main__":
    evaluate_1day_ic()
