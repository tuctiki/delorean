#!/usr/bin/env python3
"""
Generic Factor Tuning Script
Usage: python tune_factors_generic.py --start 2023-01-01 --end 2025-12-31
"""
import sys
import os
import pandas as pd
import qlib
from qlib.data import D
import argparse

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from delorean.config import QLIB_PROVIDER_URI, QLIB_REGION, ETF_LIST

def init_qlib():
    provider_uri = os.path.expanduser(QLIB_PROVIDER_URI)
    qlib.init(provider_uri=provider_uri, region=QLIB_REGION)

def tune_factor(template: str, params: list, start_time: str, end_time: str, label_horizon: int = 1):
    """
    Tune a single factor template over a list of parameters.
    """
    label_expr = f"Ref($close, -{label_horizon}) / $close - 1"
    
    variant_fields = []
    variant_names = []
    
    for p in params:
        # Handle tuple params for multi-arg templates
        if "," in str(p) and "(" in str(p): 
            # This is a bit complex for CLI, simplified to scalar for now or eval
            pass
        
        # Simple scalar substitution {W}
        name = f"Factor_{p}"
        expr = template.replace("{W}", str(p))
        variant_fields.append(expr)
        variant_names.append(name)
        
    try:
        df = D.features(ETF_LIST, variant_fields + [label_expr], start_time=start_time, end_time=end_time, freq='day')
        df.columns = variant_names + ["label"]
        df = df.dropna()
        
        results = []
        for name, p in zip(variant_names, params):
            ic = df[name].corr(df["label"], method="spearman")
            results.append({"Param": p, "IC": ic})
            
        return pd.DataFrame(results).sort_values("IC", ascending=False)
        
    except Exception as e:
        print(f"Error: {e}")
        return pd.DataFrame()

def main():
    parser = argparse.ArgumentParser(description="Tune Factor Lookback Windows")
    parser.add_argument("--template", type=str, required=True, help="Qlib expression template with {W} placeholder")
    parser.add_argument("--params", type=str, required=True, help="Comma-separated list of integers, e.g. 5,10,20")
    parser.add_argument("--start", type=str, default="2023-01-01")
    parser.add_argument("--end", type=str, default="2025-12-31")
    
    args = parser.parse_args()
    
    init_qlib()
    
    params = [int(x) for x in args.params.split(",")]
    print(f"Tuning Template: {args.template}")
    print(f"Params: {params}")
    
    results = tune_factor(args.template, params, args.start, args.end)
    print("\nResults:")
    print(results.to_string(index=False))

if __name__ == "__main__":
    main()
