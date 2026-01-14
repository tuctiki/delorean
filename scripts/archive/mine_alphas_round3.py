#!/usr/bin/env python3
"""
Round 3 Alpha Mining: Stability & Regimes
Focusing on low-turnover, regime-aware, and cross-asset signals.
"""
import sys
import os
import pandas as pd
import numpy as np
import qlib
from qlib.data import D
from qlib.config import REG_CN

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from delorean.config import QLIB_PROVIDER_URI, QLIB_REGION, ETF_LIST, BENCHMARK
from delorean.data import ETFDataHandler

def init_qlib():
    provider_uri = os.path.expanduser(QLIB_PROVIDER_URI)
    qlib.init(provider_uri=provider_uri, region=QLIB_REGION)
    print(f"Qlib initialized with provider: {provider_uri}")

def evaluate_factor(expression, instruments, start_time, end_time, label_horizon=5):
    """Evaluate factor with specified label horizon."""
    label_expr = f"Ref($close, -{label_horizon}) / $close - 1"
    fields = [expression, label_expr]
    names = ["factor", "label"]
    
    try:
        data = D.features(instruments, fields, start_time=start_time, end_time=end_time, freq='day')
        data.columns = names
        data = data.dropna()
        
        if data.empty or len(data) < 100:
            return None

        # Calculate IC
        daily_ic = data.groupby("datetime").apply(
            lambda df: df["factor"].corr(df["label"], method="spearman")
        )
        daily_ic = daily_ic.dropna()
        
        if len(daily_ic) < 10:
            return None
        
        mean_ic = daily_ic.mean()
        ic_std = daily_ic.std()
        icir = mean_ic / ic_std if ic_std != 0 else 0
        
        # Long-Only Alpha
        def get_long_alpha(df):
            try:
                top_cutoff = df["factor"].quantile(0.8)
                long_ret = df[df["factor"] >= top_cutoff]["label"].mean()
                market_ret = df["label"].mean()
                return long_ret - market_ret
            except:
                return np.nan
        
        daily_alpha = data.groupby("datetime").apply(get_long_alpha).dropna()
        annualized_alpha = daily_alpha.mean() * 252
        
        # Turnover (rank change)
        def calc_turnover(df):
            try:
                df = df.sort_index()
                rank_current = df["factor"].rank(pct=True)
                rank_prev = rank_current.shift(1)
                return (rank_current - rank_prev).abs().mean()
            except:
                return np.nan

        turnover = data.groupby(level=1).apply(calc_turnover).mean()
        
        return {
            "Expression": expression,
            "IC": mean_ic,
            "ICIR": icir,
            "Ann_Alpha_%": annualized_alpha * 100,
            "Turnover": turnover,
            "N_Days": len(daily_ic)
        }

    except Exception as e:
        # print(f"  Error: {e}")
        return None

def main():
    print("=" * 80)
    print("🧪 Round 3 Alpha Mining: Stability & Regimes")
    print("=" * 80)
    
    init_qlib()
    
    TRAIN_START = "2015-01-01"
    TRAIN_END = "2022-12-31"
    TEST_START = "2023-01-01"
    TEST_END = "2025-12-31"
    
    print(f"\n📊 Mining Period: {TRAIN_START} to {TRAIN_END}")
    print(f"✅ Validation Period: {TEST_START} to {TEST_END}")
    
    candidates = {
        # --- Stability (Low Turnover) ---
        "Stable_Mom": "Mean($close, 5) / Mean($close, 20) - 1",
        "Stable_Vol_Inv": "(Mean(Std($close, 5), 20) / Mean($close, 20)) * -1",
        "Smoothed_ROC": "Mean($close / Ref($close, 10) - 1, 5)",
        
        # --- Regime Aware ---
        # If High Vol (Bearish?), prefer Low Vol stocks. If Low Vol (Bullish?), prefer Momentum.
        # Implementation: (Vol < LongTermVol) * MOM + (Vol >= LongTermVol) * LowVol
        # Simplified for mining: Interaction terms
        "LowVol_Regime": "(Std($close, 20) / Mean(Std($close, 20), 60)) * -1", # Relative Volatility
        "Trend_Efficiency": "($close / Ref($close, 20) - 1) / (Std($close / Ref($close, 1) - 1, 20) + 0.0001)",
        
        # --- Cross Asset / Relative Strength ---
        # Note: Qlib expressions are instrument-specific. 
        # Benchmark correlation: High corr with benchmark in up-trend might be good?
        # Actually, let's look at Relative Strength vs simple History
        "RS_Rank": "Rank($close / Ref($close, 20))",
        "RS_Smoothed": "Mean(Rank($close / Ref($close, 10)), 5)",
        
        # --- Fundamental / Value Proxies (Price derived) ---
        "Value_Proxy": "($close - Min($low, 60)) / (Max($high, 60) - Min($low, 60)) * -1", # Buy Low in Range
        "Reversion_20": "($close / Mean($close, 20) - 1) * -1",
        
        # --- Volume Stability ---
        "Vol_Stability": "Std($volume, 20) / Mean($volume, 20) * -1", # Low vol of volume
        "Smart_Money": "($close - $open) / ($high - $low + 0.001) * Log($volume + 1)",
    }
    
    print(f"Testing {len(candidates)} candidates...\n")
    
    results = []
    for name, expr in candidates.items():
        print(f"Testing {name}...", end=" ", flush=True)
        res = evaluate_factor(expr, ETF_LIST, TRAIN_START, TRAIN_END)
        if res:
            res["Name"] = name
            results.append(res)
            print(f"IC: {res['IC']:.4f}, ICIR: {res['ICIR']:.2f}")
        else:
            print("Failed.")
            
    if not results:
        print("No valid factors found.")
        return

    df = pd.DataFrame(results).sort_values("IC", ascending=False)
    
    print("\n" + "=" * 80)
    print("TRAINING RESULTS")
    print("=" * 80)
    print(df[["Name", "IC", "ICIR", "Ann_Alpha_%", "Turnover"]].to_string(index=False))
    
    # Filter for OOS
    promising = df[df["ICIR"].abs() > 0.4] # Slightly relaxed for discovery
    
    if not promising.empty:
        print("\n" + "=" * 80)
        print("OUT-OF-SAMPLE VALIDATION")
        print("=" * 80)
        
        oos_results = []
        for _, row in promising.iterrows():
            name = row["Name"]
            print(f"Validating {name}...", end=" ", flush=True)
            res = evaluate_factor(row["Expression"], ETF_LIST, TEST_START, TEST_END)
            if res:
                res["Name"] = name
                res["Train_IC"] = row["IC"]
                res["Train_ICIR"] = row["ICIR"]
                oos_results.append(res)
                print(f"OOS IC: {res['IC']:.4f}")
            else:
                print("Failed.")
        
        if oos_results:
            df_oos = pd.DataFrame(oos_results).sort_values("IC", ascending=False)
            print("\n")
            print(df_oos[["Name", "Train_IC", "IC", "Train_ICIR", "ICIR", "Ann_Alpha_%"]].to_string(index=False))
            
            # Save
            df_oos.to_csv("artifacts/round3_results.csv", index=False)
            print("\nSaved to artifacts/round3_results.csv")

if __name__ == "__main__":
    main()
