#!/usr/bin/env python3
"""
Validate Top Performers on Out-of-Sample Data
"""
import sys
import os
import pandas as pd
import numpy as np
import qlib
from qlib.data import D

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from delorean.config import QLIB_PROVIDER_URI, QLIB_REGION, ETF_LIST
from delorean.data import ETFDataHandler

def init_qlib():
    provider_uri = os.path.expanduser(QLIB_PROVIDER_URI)
    qlib.init(provider_uri=provider_uri, region=QLIB_REGION)

def evaluate_factor(expression, instruments, start_time, end_time, label_horizon=5):
    label_expr = f"Ref($close, -{label_horizon}) / $close - 1"
    fields = [expression, label_expr]
    names = ["factor", "label"]
    
    try:
        data = D.features(instruments, fields, start_time=start_time, end_time=end_time, freq='day')
        data.columns = names
        data = data.dropna()
        
        if data.empty or len(data) < 50:
            return None

        daily_ic = data.groupby("datetime").apply(
            lambda df: df["factor"].corr(df["label"], method="spearman")
        )
        daily_ic = daily_ic.dropna()
        
        if len(daily_ic) < 5:
            return None
        
        mean_ic = daily_ic.mean()
        ic_std = daily_ic.std()
        icir = mean_ic / ic_std if ic_std != 0 else 0
        
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
        alpha_sharpe = (daily_alpha.mean() / daily_alpha.std() * np.sqrt(252)) if daily_alpha.std() != 0 else 0
        
        return {
            "IC": mean_ic,
            "ICIR": icir,
            "Ann_Alpha_%": annualized_alpha * 100,
            "Alpha_Sharpe": alpha_sharpe,
            "N_Days": len(daily_ic)
        }
    except Exception as e:
        return None

def calculate_correlation_with_existing(new_expr, instruments, start_time, end_time):
    """Calculate correlation with existing factors."""
    try:
        existing_exprs, existing_names = ETFDataHandler.get_custom_factors()
        
        new_data = D.features(instruments, [new_expr], start_time=start_time, end_time=end_time, freq='day')
        new_data.columns = ["new_factor"]
        new_data = new_data.dropna()
        
        correlations = {}
        for i, (existing_expr, existing_name) in enumerate(zip(existing_exprs, existing_names)):
            try:
                existing_data = D.features(instruments, [existing_expr], start_time=start_time, end_time=end_time, freq='day')
                existing_data.columns = ["existing_factor"]
                existing_data = existing_data.dropna()
                
                merged = new_data.join(existing_data, how='inner')
                if len(merged) > 100:
                    corr = merged["new_factor"].corr(merged["existing_factor"])
                    correlations[existing_name] = corr
            except:
                continue
        
        return correlations
    except:
        return {}

def main():
    print("=" * 80)
    print("🔍 Top Factor Validation & Correlation Analysis")
    print("=" * 80)
    
    init_qlib()
    
    TRAIN_START = "2015-01-01"
    TRAIN_END = "2022-12-31"
    TEST_START = "2023-01-01"
    TEST_END = "2025-12-31"
    
    # Top 6 performers from Round 2
    top_factors = {
        "VolAdj_Mom": "($close / Ref($close, 10) - 1) / (Std($close / Ref($close, 1) - 1, 10) + 0.001)",
        "Mom_Vol_Combo": "($close / Ref($close, 10) - 1) * (1 / (Std($close / Ref($close, 1) - 1, 20) + 0.001))",
        "Price_Impact": "Abs($close / Ref($close, 1) - 1) / (Log($volume + 1) + 0.001)",
        "Gap_Size": "Abs($open / Ref($close, 1) - 1)",
        "Range_Ratio": "($high - $low) / ($close + 0.001)",
        "Gap_Fill": "($close - $open) / (Abs($open - Ref($close, 1)) + 0.001)",
    }
    
    print(f"\n📊 Validating {len(top_factors)} top factors...")
    print(f"Training: {TRAIN_START} to {TRAIN_END}")
    print(f"Testing:  {TEST_START} to {TEST_END}\n")
    
    results = []
    for name, expr in top_factors.items():
        print(f"🔬 {name}")
        
        # Training metrics
        train_res = evaluate_factor(expr, ETF_LIST, TRAIN_START, TRAIN_END, label_horizon=5)
        
        # Testing metrics
        test_res = evaluate_factor(expr, ETF_LIST, TEST_START, TEST_END, label_horizon=5)
        
        # Correlation with existing
        corrs = calculate_correlation_with_existing(expr, ETF_LIST, TRAIN_START, TRAIN_END)
        max_corr = max(abs(v) for v in corrs.values()) if corrs else 0
        
        if train_res and test_res:
            results.append({
                "Factor_Name": name,
                "Train_IC": train_res["IC"],
                "Test_IC": test_res["IC"],
                "Train_ICIR": train_res["ICIR"],
                "Test_ICIR": test_res["ICIR"],
                "Train_Alpha_%": train_res["Ann_Alpha_%"],
                "Test_Alpha_%": test_res["Ann_Alpha_%"],
                "Test_Sharpe": test_res["Alpha_Sharpe"],
                "Max_Corr_Existing": max_corr,
                "IC_Consistent": (train_res["IC"] > 0 and test_res["IC"] > 0) or (train_res["IC"] < 0 and test_res["IC"] < 0),
                "Expression": expr
            })
            
            print(f"  Train: IC={train_res['IC']:.4f}, ICIR={train_res['ICIR']:.2f}, Alpha={train_res['Ann_Alpha_%']:.1f}%")
            print(f"  Test:  IC={test_res['IC']:.4f}, ICIR={test_res['ICIR']:.2f}, Alpha={test_res['Ann_Alpha_%']:.1f}%")
            print(f"  Max Corr w/ Existing: {max_corr:.3f}")
            print(f"  Consistent: {'✓' if results[-1]['IC_Consistent'] else '✗'}\n")
    
    df = pd.DataFrame(results)
    df = df.sort_values("Test_IC", ascending=False)
    
    print("\n" + "=" * 80)
    print("📈 VALIDATION SUMMARY")
    print("=" * 80)
    
    display_cols = ["Factor_Name", "Train_IC", "Test_IC", "Train_ICIR", "Test_ICIR", 
                    "Test_Alpha_%", "Test_Sharpe", "Max_Corr_Existing", "IC_Consistent"]
    print(df[display_cols].to_string(index=False))
    
    # Filter for integration candidates
    candidates = df[
        (df["IC_Consistent"] == True) &
        (df["Test_IC"].abs() > 0.015) &
        (df["Max_Corr_Existing"] < 0.7)
    ]
    
    print(f"\n" + "=" * 80)
    print(f"✅ INTEGRATION CANDIDATES (Consistent, |Test IC| > 0.015, Max Corr < 0.7)")
    print("=" * 80)
    print(f"Found {len(candidates)} factors:\n")
    
    if len(candidates) > 0:
        for _, row in candidates.iterrows():
            print(f"📌 {row['Factor_Name']}")
            print(f"   Formula: {row['Expression']}")
            print(f"   Test IC: {row['Test_IC']:.4f}, ICIR: {row['Test_ICIR']:.2f}")
            print(f"   Test Alpha: {row['Test_Alpha_%']:.1f}%, Sharpe: {row['Test_Sharpe']:.2f}")
            print(f"   Max Correlation: {row['Max_Corr_Existing']:.3f}\n")
    
    # Save results
    df.to_csv("artifacts/alpha_validation_summary.csv", index=False)
    print(f"💾 Results saved to artifacts/alpha_validation_summary.csv\n")
    
    print("=" * 80)
    print("✅ Validation Complete!")
    print("=" * 80)

if __name__ == "__main__":
    main()
