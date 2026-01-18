
import qlib
import pandas as pd
import numpy as np
from qlib.data import D
import os
import sys

# Setup path
sys.path.insert(0, os.getcwd())

from delorean.conf import QLIB_PROVIDER_URI, QLIB_REGION, ETF_LIST

def audit_hybrid_v1():
    qlib.init(provider_uri=QLIB_PROVIDER_URI, region=QLIB_REGION)
    
    factors = {
        "Alpha_Gen_8": "-1 * (Sum(-1 * (Log($open + 1e-5)), 5) + Std($high, 5))",
        "Smart_Flow_Rev": "-1 * ($close - $low) / ($high - $low + 0.001) * (Mean($volume, 5) / Mean($volume, 20))",
        "Structural_Breakout": "($close - Min($low, 60)) / (Max($high, 60) - Min($low, 60) + 1e-4)",
        "MOM120": "$close / Ref($close, 120) - 1",
        "Vol_Price_Div_Rev": "Mean(-1 * Corr($close / Ref($close, 1), $volume / Ref($volume, 1), 10), 5)",
        "Vol_Skew_20": "-1 * Skew($close / Ref($close, 1) - 1, 20)",
    }
    
    label_expr = "Ref($close, -1) / $close - 1"
    
    start_date = "2023-01-01"
    end_date = "2026-01-17"
    
    exprs = list(factors.values())
    names = list(factors.keys())
    
    df = D.features(ETF_LIST, exprs + [label_expr], start_time=start_date, end_time=end_date)
    df.columns = names + ["label"]
    df = df.dropna()
    
    print(f"\n--- Hybrid V1 IC (2023-Recent) ---")
    for name in names:
        ic = df[name].corr(df["label"], method="spearman")
        print(f"{name:<20}: IC = {ic:.4f}")
        
    print(f"\n--- Correlation Matrix ---")
    corr = df[names].corr(method="spearman")
    print(corr)

if __name__ == "__main__":
    audit_hybrid_v1()
