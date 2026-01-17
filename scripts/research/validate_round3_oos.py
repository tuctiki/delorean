#!/usr/bin/env python3
"""
Validate Top Round 3 Factors OOS
"""
import sys
import os
import pandas as pd
import numpy as np
import qlib
from qlib.data import D
from qlib.config import REG_CN

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from delorean.config import QLIB_PROVIDER_URI, QLIB_REGION, ETF_LIST

def init_qlib():
    provider_uri = os.path.expanduser(QLIB_PROVIDER_URI)
    qlib.init(provider_uri=provider_uri, region=QLIB_REGION)

def evaluate(name, expr, start, end):
    print(f"Validating {name} ({start} to {end})...")
    label_expr = "Ref($close, -5) / $close - 1"
    try:
        data = D.features(ETF_LIST, [expr, label_expr], start_time=start, end_time=end, freq='day')
        data.columns = ["factor", "label"]
        data = data.dropna()
        
        if data.empty:
            print("  No data")
            return

        ic = data.groupby("datetime").apply(lambda df: df["factor"].corr(df["label"], method="spearman"))
        ic_mean = ic.mean()
        ic_std = ic.std()
        icir = ic_mean / ic_std if ic_std != 0 else 0
        
        print(f"  IC: {ic_mean:.4f}")
        print(f"  ICIR: {icir:.4f}")
        
    except Exception as e:
        print(f"  Error: {e}")

def main():
    init_qlib()
    
    factors = {
        "Stable_Mom": "Mean($close, 5) / Mean($close, 20) - 1",
        "Smoothed_ROC": "Mean($close / Ref($close, 10) - 1, 5)",
        "Trend_Efficiency": "($close / Ref($close, 20) - 1) / (Std($close / Ref($close, 1) - 1, 20) + 0.0001)"
    }
    
    TEST_START = "2023-01-01"
    TEST_END = "2025-12-31"
    
    for name, expr in factors.items():
        evaluate(name, expr, TEST_START, TEST_END)
        print("-" * 20)

if __name__ == "__main__":
    main()
