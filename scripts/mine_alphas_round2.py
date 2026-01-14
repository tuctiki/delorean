#!/usr/bin/env python3
"""
Refined Alpha Mining - Round 2
Testing refined formulations and alternative label horizons.
"""
import sys
import os
import pandas as pd
import numpy as np
import qlib
from qlib.data import D
from qlib.config import REG_CN

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from delorean.config import QLIB_PROVIDER_URI, QLIB_REGION, ETF_LIST
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
        nan_ic_rate = daily_ic.isna().mean()
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
        alpha_sharpe = (daily_alpha.mean() / daily_alpha.std() * np.sqrt(252)) if daily_alpha.std() != 0 else 0
        
        return {
            "Expression": expression,
            "IC": mean_ic,
            "IC_Std": ic_std,
            "ICIR": icir,
            "Ann_Alpha_%": annualized_alpha * 100,
            "Alpha_Sharpe": alpha_sharpe,
            "NaN_IC_Rate_%": nan_ic_rate * 100,
            "N_Days": len(daily_ic)
        }

    except Exception as e:
        print(f"  ❌ Error: {e}")
        return None

def main():
    print("=" * 80)
    print("🔬 Refined Alpha Mining - Round 2 (5-Day Horizon)")
    print("=" * 80)
    
    init_qlib()
    
    TRAIN_START = "2015-01-01"
    TRAIN_END = "2022-12-31"
    TEST_START = "2023-01-01"
    TEST_END = "2025-12-31"
    
    print(f"\n📊 Mining Period: {TRAIN_START} to {TRAIN_END}")
    print(f"✅ Validation Period: {TEST_START} to {TEST_END}")
    print(f"🎯 Label: 5-Day Forward Return (Production Setting)\n")
    
    # Refined candidates with better formulations
    candidates = {
        # Volume-based signals
        "Vol_Exhaustion": "Log($volume / Mean($volume, 20) + 1) * -1",
        "Vol_Price_Div": "Corr($close / Ref($close, 1) - 1, $volume / Ref($volume, 1) - 1, 10) * -1",
        "Vol_Spike_Down": "If($close / Ref($close, 1) - 1 < -0.02, $volume / Mean($volume, 20), 0)",
        
        # Momentum & Reversal
        "Short_Rev": "($close / Ref($close, 3) - 1) * -1",
        "Mom_Decel": "($close / Ref($close, 5) - 1) - ($close / Ref($close, 10) - 1)",
        "VolAdj_Mom": "($close / Ref($close, 10) - 1) / (Std($close / Ref($close, 1) - 1, 10) + 0.001)",
        "ROC_Ratio": "($close / Ref($close, 5)) / ($close / Ref($close, 20) + 0.001)",
        
        # Volatility & Range
        "Vol_Compression": "(Std($close / Ref($close, 1) - 1, 5) / Std($close / Ref($close, 1) - 1, 20)) * -1",
        "Range_Ratio": "($high - $low) / ($close + 0.001)",
        "Close_Position": "2 * ($close - ($high + $low) / 2) / ($high - $low + 0.001)",
        
        # Cross-sectional (using CSRank which doesn't need N parameter)
        "Mom20_Rank": "CSRank($close / Ref($close, 20) - 1)",
        "Vol20_Rank_Inv": "CSRank(Std($close / Ref($close, 1) - 1, 20)) * -1",
        
        # Liquidity & Stress
        "Turnover_Shock": "(($volume * $close) / Ref($volume * $close, 5) - 1) * -1",
        "Price_Impact": "Abs($close / Ref($close, 1) - 1) / (Log($volume + 1) + 0.001)",
        
        # Gap & Overnight
        "Gap_Size": "Abs($open / Ref($close, 1) - 1)",
        "Gap_Fill": "($close - $open) / (Abs($open - Ref($close, 1)) + 0.001)",
        
        # Composite signals
        "Mom_Vol_Combo": "($close / Ref($close, 10) - 1) * (1 / (Std($close / Ref($close, 1) - 1, 20) + 0.001))",
        "Rev_Vol_Combo": "($close / Ref($close, 5) - 1) * -1 * (1 / (Std($close / Ref($close, 1) - 1, 10) + 0.001))",
    }
    
    print(f"🧪 Testing {len(candidates)} refined factor candidates...\n")
    
    # Evaluate on Training Period (5-day horizon)
    results_train = []
    for name, expr in candidates.items():
        print(f"📈 {name}:")
        print(f"   Formula: {expr[:80]}{'...' if len(expr) > 80 else ''}")
        res = evaluate_factor(expr, ETF_LIST, TRAIN_START, TRAIN_END, label_horizon=5)
        if res:
            res["Factor_Name"] = name
            results_train.append(res)
            print(f"   ✓ IC={res['IC']:.4f}, ICIR={res['ICIR']:.2f}, Alpha={res['Ann_Alpha_%']:.2f}%")
        print()
    
    if not results_train:
        print("❌ No valid results found.")
        return
    
    df_train = pd.DataFrame(results_train)
    df_train = df_train.sort_values("IC", ascending=False)
    
    cols = ["Factor_Name", "IC", "ICIR", "Ann_Alpha_%", "Alpha_Sharpe", "IC_Std", "NaN_IC_Rate_%", "N_Days", "Expression"]
    df_train = df_train[cols]
    
    print("\n" + "=" * 80)
    print("📊 TRAINING PERIOD RESULTS (2015-2022, H=5)")
    print("=" * 80)
    print(df_train.to_string(index=False))
    
    # Relaxed thresholds for ETF universe (smaller, less liquid)
    top_factors = df_train[
        (df_train["IC"].abs() > 0.02) & 
        (df_train["ICIR"].abs() > 0.3)
    ]
    
    print(f"\n🎯 Promising Factors (|IC| > 0.02, |ICIR| > 0.3):")
    print(f"   Found {len(top_factors)} factors\n")
    
    if len(top_factors) > 0:
        print(top_factors[["Factor_Name", "IC", "ICIR", "Ann_Alpha_%"]].to_string(index=False))
        
        # Out-of-Sample Validation
        print(f"\n" + "=" * 80)
        print("🔍 OUT-OF-SAMPLE VALIDATION (2023-Present, H=5)")
        print("=" * 80)
        
        results_test = []
        for _, row in top_factors.iterrows():
            name = row["Factor_Name"]
            expr = row["Expression"]
            print(f"\n📊 Validating {name}...")
            res = evaluate_factor(expr, ETF_LIST, TEST_START, TEST_END, label_horizon=5)
            if res:
                res["Factor_Name"] = name
                res["Train_IC"] = row["IC"]
                res["Train_ICIR"] = row["ICIR"]
                results_test.append(res)
                print(f"   Train: IC={row['IC']:.4f}, ICIR={row['ICIR']:.2f}")
                print(f"   OOS:   IC={res['IC']:.4f}, ICIR={res['ICIR']:.2f}, Alpha={res['Ann_Alpha_%']:.2f}%")
        
        if results_test:
            df_test = pd.DataFrame(results_test)
            df_test = df_test.sort_values("IC", ascending=False)
            
            test_cols = ["Factor_Name", "Train_IC", "IC", "Train_ICIR", "ICIR", "Ann_Alpha_%", "Alpha_Sharpe", "Expression"]
            df_test = df_test[test_cols]
            
            print("\n" + "=" * 80)
            print("📈 OUT-OF-SAMPLE SUMMARY")
            print("=" * 80)
            print(df_test.to_string(index=False))
            
            # Check for consistency (same sign IC)
            consistent = df_test[
                ((df_test["Train_IC"] > 0) & (df_test["IC"] > 0)) |
                ((df_test["Train_IC"] < 0) & (df_test["IC"] < 0))
            ]
            
            print(f"\n✅ Consistent Factors (same IC sign in/out-of-sample): {len(consistent)}")
            if len(consistent) > 0:
                print(consistent[["Factor_Name", "Train_IC", "IC", "Ann_Alpha_%"]].to_string(index=False))
    
    # Save results
    output_file = "artifacts/alpha_mining_round2_h5.csv"
    df_train.to_csv(output_file, index=False)
    print(f"\n💾 Results saved to {output_file}")
    
    if len(top_factors) > 0 and results_test:
        oos_file = "artifacts/alpha_mining_round2_oos.csv"
        df_test.to_csv(oos_file, index=False)
        print(f"💾 OOS validation saved to {oos_file}")
    
    print("\n" + "=" * 80)
    print("✅ Refined Alpha Mining Complete!")
    print("=" * 80)

if __name__ == "__main__":
    main()
