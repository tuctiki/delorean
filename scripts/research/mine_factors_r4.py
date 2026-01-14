#!/usr/bin/env python3
"""
Round 4 Alpha Mining: New Signal Discovery
Focus: Volume-Price Dynamics, Momentum Variants, Mean Reversion, Cross-Sectional

Train: 2015-01-01 to 2022-12-31
Test:  2023-01-01 to 2025-12-31
"""
import sys
import os
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import qlib
from qlib.data import D
from delorean.config import QLIB_PROVIDER_URI, QLIB_REGION, ETF_LIST
from delorean.data import ETFDataHandler


def init_qlib():
    provider_uri = os.path.expanduser(QLIB_PROVIDER_URI)
    qlib.init(provider_uri=provider_uri, region=QLIB_REGION)
    print(f"Qlib initialized: {provider_uri}")


def evaluate_factor(expression: str, instruments: list, start_time: str, end_time: str, 
                    label_horizon: int = 5) -> dict | None:
    """Evaluate a factor expression and return metrics."""
    label_expr = f"Ref($close, -{label_horizon}) / $close - 1"
    
    try:
        data = D.features(instruments, [expression, label_expr], 
                         start_time=start_time, end_time=end_time, freq='day')
        data.columns = ["factor", "label"]
        data = data.dropna()
        
        if data.empty or len(data) < 100:
            return None
        
        # Daily IC (Spearman)
        daily_ic = data.groupby("datetime").apply(
            lambda df: df["factor"].corr(df["label"], method="spearman")
        ).dropna()
        
        if len(daily_ic) < 10:
            return None
        
        mean_ic = daily_ic.mean()
        ic_std = daily_ic.std()
        icir = mean_ic / ic_std if ic_std != 0 else 0
        
        # Long-only alpha (top quintile vs market)
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
            "Alpha_%": annualized_alpha * 100,
            "Turnover": turnover,
            "N": len(daily_ic)
        }
        
    except Exception as e:
        return None


def get_existing_factors():
    """Get existing factor expressions for correlation check."""
    exprs, names = ETFDataHandler.get_custom_factors()
    return dict(zip(names, exprs))


def check_correlation(new_expr: str, existing_factors: dict, instruments: list,
                      start_time: str, end_time: str) -> dict:
    """Check correlation of new factor with existing factors."""
    all_exprs = [new_expr] + list(existing_factors.values())
    all_names = ["NEW"] + list(existing_factors.keys())
    
    try:
        data = D.features(instruments, all_exprs, start_time=start_time, end_time=end_time)
        data.columns = all_names
        data = data.dropna()
        
        if data.empty:
            return {}
        
        # Cross-sectional correlation per day, then average
        def daily_corr(df):
            return df.corr()["NEW"].drop("NEW")
        
        corr_series = data.groupby("datetime").apply(daily_corr).mean()
        return corr_series.to_dict()
        
    except Exception as e:
        return {}


def main():
    print("=" * 80)
    print("🔬 Round 4 Alpha Mining: New Signal Discovery")
    print("=" * 80)
    
    init_qlib()
    
    TRAIN_START = "2015-01-01"
    TRAIN_END = "2022-12-31"
    TEST_START = "2023-01-01"
    TEST_END = "2025-12-31"
    
    print(f"\n📊 Train: {TRAIN_START} to {TRAIN_END}")
    print(f"✅ Test:  {TEST_START} to {TEST_END}")
    
    # Factor Candidates (Hypothesis-Driven)
    candidates = {
        # --- Volume-Price Dynamics ---
        "Volume_Breakout": "$volume / Mean($volume, 20) * ($close / Ref($close, 1) - 1)",
        "Smart_Flow": "($close - $low) / ($high - $low + 0.001) * (Mean($volume, 5) / Mean($volume, 20))",
        "VWAP_Dev": "($close - Mean($close * $volume, 5) / Mean($volume, 5)) / $close",
        
        # --- Momentum Variants ---
        "Mom_Persistence": "Sum(If($close > Ref($close, 1), 1, 0), 10) / 10",
        "Channel_Breakout": "($close - Min($low, 20)) / (Max($high, 20) - Min($low, 20) + 0.001)",
        "Acceleration": "($close / Ref($close, 5) - 1) - (Ref($close, 5) / Ref($close, 10) - 1)",
        
        # --- Mean Reversion ---
        "Oversold_Signal": "(Mean($close, 5) / Mean($close, 20) - 1) * -1",
        "RSI_Proxy": "Sum(If($close > Ref($close, 1), $close - Ref($close, 1), 0), 14) / (Sum(Abs($close - Ref($close, 1)), 14) + 0.001)",
        "BB_Reversion": "(Mean($close, 20) - $close) / (Std($close, 20) + 0.001)",
        
        # --- Cross-Sectional / Relative ---
        "RS_Momentum": "Rank($close / Ref($close, 20))",
        "RS_Smooth": "Mean(Rank($close / Ref($close, 10)), 5)",
        "Sector_Leader": "Rank(Mean($close / Ref($close, 5), 3))",
        
        # --- Volatility Adjusted ---
        "Realized_Vol_Ratio": "Std($close / Ref($close, 1) - 1, 10) / Std($close / Ref($close, 1) - 1, 60)",
        "Vol_Regime": "(Std($close, 20) / Mean(Std($close, 20), 60)) * -1",
    }
    
    print(f"\n🧪 Testing {len(candidates)} candidates...\n")
    
    # Phase 1: Training Evaluation
    print("-" * 60)
    print("PHASE 1: TRAINING PERIOD EVALUATION")
    print("-" * 60)
    
    results = []
    for name, expr in candidates.items():
        print(f"  {name}...", end=" ", flush=True)
        res = evaluate_factor(expr, ETF_LIST, TRAIN_START, TRAIN_END)
        if res:
            res["Name"] = name
            results.append(res)
            status = "✓" if res["IC"] > 0.02 else "○"
            print(f"{status} IC: {res['IC']:.4f}, ICIR: {res['ICIR']:.2f}")
        else:
            print("✗ Failed")
    
    if not results:
        print("\n❌ No valid factors found in training.")
        return
    
    df_train = pd.DataFrame(results).sort_values("IC", ascending=False)
    
    print("\n" + "=" * 60)
    print("TRAINING RESULTS (sorted by IC)")
    print("=" * 60)
    print(df_train[["Name", "IC", "ICIR", "Alpha_%", "Turnover"]].to_string(index=False))
    
    # Filter promising candidates (IC > 0.02 or ICIR > 0.4)
    promising = df_train[(df_train["IC"].abs() > 0.02) | (df_train["ICIR"].abs() > 0.4)]
    
    if promising.empty:
        print("\n⚠️ No factors passed training threshold (IC > 0.02 or ICIR > 0.4)")
        return
    
    # Phase 2: Out-of-Sample Validation
    print("\n" + "-" * 60)
    print("PHASE 2: OUT-OF-SAMPLE VALIDATION")
    print("-" * 60)
    
    oos_results = []
    for _, row in promising.iterrows():
        name = row["Name"]
        expr = row["Expression"]
        print(f"  {name}...", end=" ", flush=True)
        
        res = evaluate_factor(expr, ETF_LIST, TEST_START, TEST_END)
        if res:
            res["Name"] = name
            res["Train_IC"] = row["IC"]
            res["Train_ICIR"] = row["ICIR"]
            res["IC_Decay"] = (row["IC"] - res["IC"]) / abs(row["IC"]) if row["IC"] != 0 else 0
            oos_results.append(res)
            
            status = "✓" if res["IC"] > 0.02 else "○"
            print(f"{status} OOS IC: {res['IC']:.4f} (decay: {res['IC_Decay']*100:.1f}%)")
        else:
            print("✗ Failed")
    
    if not oos_results:
        print("\n❌ No factors passed OOS validation.")
        return
    
    df_oos = pd.DataFrame(oos_results).sort_values("IC", ascending=False)
    
    print("\n" + "=" * 60)
    print("OUT-OF-SAMPLE RESULTS")
    print("=" * 60)
    print(df_oos[["Name", "Train_IC", "IC", "ICIR", "Alpha_%", "IC_Decay"]].to_string(index=False))
    
    # Phase 3: Correlation Check
    print("\n" + "-" * 60)
    print("PHASE 3: CORRELATION WITH EXISTING FACTORS")
    print("-" * 60)
    
    existing = get_existing_factors()
    validated = df_oos[df_oos["IC"] > 0.02]
    
    final_candidates = []
    for _, row in validated.iterrows():
        name = row["Name"]
        expr = row["Expression"]
        print(f"  {name}:", end=" ", flush=True)
        
        corr = check_correlation(expr, existing, ETF_LIST, TRAIN_START, TRAIN_END)
        if corr:
            max_corr_name = max(corr, key=lambda k: abs(corr[k]))
            max_corr_val = corr[max_corr_name]
            
            if abs(max_corr_val) < 0.7:
                print(f"✓ Max corr: {max_corr_val:.2f} with {max_corr_name}")
                row["Max_Corr"] = max_corr_val
                row["Corr_With"] = max_corr_name
                final_candidates.append(row)
            else:
                print(f"✗ Too correlated: {max_corr_val:.2f} with {max_corr_name}")
        else:
            print("⚠️ Could not compute correlation")
            final_candidates.append(row)
    
    # Final Summary
    print("\n" + "=" * 80)
    print("🏆 FINAL CANDIDATES FOR INTEGRATION")
    print("=" * 80)
    
    if final_candidates:
        df_final = pd.DataFrame(final_candidates)
        print(df_final[["Name", "Expression", "IC", "ICIR", "Alpha_%"]].to_string(index=False))
        
        # Save results
        os.makedirs("artifacts", exist_ok=True)
        df_final.to_csv("artifacts/round4_candidates.csv", index=False)
        print("\n📁 Saved to artifacts/round4_candidates.csv")
        
        print("\n📋 To add a factor to production, update delorean/data.py:")
        for _, row in df_final.iterrows():
            print(f'  "{row["Expression"]}",  # {row["Name"]} (IC={row["IC"]:.3f})')
    else:
        print("❌ No factors passed all validation criteria.")
        print("Consider relaxing thresholds or generating new hypotheses.")


if __name__ == "__main__":
    main()
