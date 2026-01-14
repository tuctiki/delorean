#!/usr/bin/env python3
"""
Enhanced Alpha Mining Script
Discovers and validates new alpha factors for ETF trading strategy.
"""
import sys
import os
import pandas as pd
import numpy as np
import qlib
from qlib.data import D
from qlib.config import REG_CN
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from delorean.config import QLIB_PROVIDER_URI, QLIB_REGION, ETF_LIST, START_TIME, END_TIME
from delorean.data import ETFDataHandler

def init_qlib():
    """Initialize Qlib with ETF data provider."""
    provider_uri = os.path.expanduser(QLIB_PROVIDER_URI)
    qlib.init(provider_uri=provider_uri, region=QLIB_REGION)
    print(f"Qlib initialized with provider: {provider_uri}")

def evaluate_factor(expression, instruments, start_time, end_time, label_horizon=1):
    """
    Evaluates a single factor expression.
    
    Args:
        expression: Factor formula (Qlib expression)
        instruments: List of ETF codes
        start_time: Start date (YYYY-MM-DD)
        end_time: End date (YYYY-MM-DD)
        label_horizon: Forward return horizon in days
    
    Returns:
        Dictionary with evaluation metrics or None if failed
    """
    # Define label based on horizon
    label_expr = f"Ref($close, -{label_horizon}) / $close - 1"
    fields = [expression, label_expr]
    names = ["factor", "label"]
    
    try:
        # Load data
        data = D.features(
            instruments, 
            fields, 
            start_time=start_time, 
            end_time=end_time, 
            freq='day'
        )
        data.columns = names
        data = data.dropna()
        
        if data.empty or len(data) < 100:
            print(f"  ⚠️  Insufficient data: {len(data)} rows")
            return None

        # Calculate IC (Information Coefficient) - Spearman Rank Correlation
        daily_ic = data.groupby("datetime").apply(
            lambda df: df["factor"].corr(df["label"], method="spearman")
        )
        nan_ic_rate = daily_ic.isna().mean()
        daily_ic = daily_ic.dropna()
        
        if len(daily_ic) < 10:
            print(f"  ⚠️  Too few valid IC values: {len(daily_ic)}")
            return None
        
        mean_ic = daily_ic.mean()
        ic_std = daily_ic.std()
        icir = mean_ic / ic_std if ic_std != 0 else 0
        
        # Calculate Long-Only Alpha (Top 20% vs Market Mean)
        def get_long_alpha(df):
            try:
                top_cutoff = df["factor"].quantile(0.8)
                long_ret = df[df["factor"] >= top_cutoff]["label"].mean()
                market_ret = df["label"].mean()
                return long_ret - market_ret
            except:
                return np.nan
        
        daily_alpha = data.groupby("datetime").apply(get_long_alpha)
        daily_alpha = daily_alpha.dropna()
        annualized_alpha = daily_alpha.mean() * 252
        alpha_sharpe = (daily_alpha.mean() / daily_alpha.std() * np.sqrt(252)) if daily_alpha.std() != 0 else 0
        
        # Calculate Turnover (factor rank change)
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
            "IC_Std": ic_std,
            "ICIR": icir,
            "Ann_Alpha_%": annualized_alpha * 100,
            "Alpha_Sharpe": alpha_sharpe,
            "Turnover": turnover,
            "NaN_IC_Rate_%": nan_ic_rate * 100,
            "N_Days": len(daily_ic)
        }

    except Exception as e:
        print(f"  ❌ Error: {e}")
        return None

def get_existing_factors():
    """Get existing factors from ETFDataHandler for correlation check."""
    custom_exprs, custom_names = ETFDataHandler.get_custom_factors()
    return custom_exprs, custom_names

def calculate_factor_correlation(new_expr, existing_exprs, instruments, start_time, end_time):
    """
    Calculate correlation between a new factor and existing factors.
    
    Returns:
        Dictionary mapping existing factor names to correlation values
    """
    try:
        # Load new factor
        new_data = D.features(
            instruments,
            [new_expr],
            start_time=start_time,
            end_time=end_time,
            freq='day'
        )
        new_data.columns = ["new_factor"]
        new_data = new_data.dropna()
        
        correlations = {}
        for i, existing_expr in enumerate(existing_exprs):
            try:
                existing_data = D.features(
                    instruments,
                    [existing_expr],
                    start_time=start_time,
                    end_time=end_time,
                    freq='day'
                )
                existing_data.columns = ["existing_factor"]
                existing_data = existing_data.dropna()
                
                # Merge on datetime and instrument
                merged = new_data.join(existing_data, how='inner')
                if len(merged) > 100:
                    corr = merged["new_factor"].corr(merged["existing_factor"])
                    correlations[f"Factor_{i+1}"] = corr
            except:
                continue
        
        return correlations
    except:
        return {}

def main():
    print("=" * 80)
    print("🔬 Enhanced Alpha Mining - Delorean ETF Strategy")
    print("=" * 80)
    
    init_qlib()
    
    # Mining Period: 2015-2022 (Training)
    TRAIN_START = "2015-01-01"
    TRAIN_END = "2022-12-31"
    
    # Validation Period: 2023-Present (Out-of-Sample)
    TEST_START = "2023-01-01"
    TEST_END = "2025-12-31"
    
    print(f"\n📊 Mining Period: {TRAIN_START} to {TRAIN_END}")
    print(f"✅ Validation Period: {TEST_START} to {TEST_END}")
    print(f"🎯 Universe: {len(ETF_LIST)} ETFs\n")
    
    # New Alpha Factor Candidates
    candidates = {
        # Category 1: Volume-Price Dynamics
        "Vol_Shock_Rev": "(-1 * ($close - Ref($close, 1))) * (($volume - Mean($volume, 20)) / (Std($volume, 20) + 0.0001))",
        "Price_Vol_Corr": "Corr($close, $volume, 20)",
        "Vol_Momentum_Rev": "($volume / Ref($volume, 5) - 1) * -1",
        
        # Category 2: Intraday Range & Volatility
        "True_Range_Norm": "Mean(Max(Max($high - $low, Abs($high - Ref($close, 1))), Abs($low - Ref($close, 1))), 10) / ($close + 0.0001)",
        "HL_Position": "($close - $low) / ($high - $low + 0.0001)",
        "Gap_Reversal": "($open - Ref($close, 1)) / (Ref($close, 1) + 0.0001) * -1",
        
        # Category 3: Advanced Momentum & Reversal
        "Mom_Divergence": "($close / Ref($close, 5) - 1) - ($close / Ref($close, 20) - 1)",
        "VolAdj_Rev": "(-1 * ($close / Ref($close, 5) - 1)) / (Std($close / Ref($close, 1) - 1, 20) + 0.0001)",
        "Accel_Ratio": "(($close - Ref($close, 5)) - (Ref($close, 5) - Ref($close, 10))) / (Ref($close, 10) + 0.0001)",
        
        # Category 4: Cross-Sectional & Relative Strength
        "Rel_Strength": "Rank($close / Ref($close, 20))",
        "Vol_Rank_Inv": "Rank(Std($close / Ref($close, 1) - 1, 20)) * -1",
        
        # Category 5: Regime Detection
        "Vol_Regime_Shift": "(Std($close / Ref($close, 1) - 1, 5) / (Std($close / Ref($close, 1) - 1, 60) + 0.0001)) * -1",
        "Trend_Strength": "Abs($close - Mean($close, 20)) / (Std($close, 20) + 0.0001)",
        "Liquidity_Stress": "Std($volume, 5) / (Mean($volume, 20) + 0.0001)",
    }
    
    print(f"🧪 Testing {len(candidates)} new factor candidates...\n")
    
    # Evaluate on Training Period
    results_train = []
    for name, expr in candidates.items():
        print(f"📈 {name}:")
        print(f"   Formula: {expr[:80]}{'...' if len(expr) > 80 else ''}")
        res = evaluate_factor(expr, ETF_LIST, TRAIN_START, TRAIN_END, label_horizon=1)
        if res:
            res["Factor_Name"] = name
            results_train.append(res)
            print(f"   ✓ IC={res['IC']:.4f}, ICIR={res['ICIR']:.2f}, Alpha={res['Ann_Alpha_%']:.2f}%")
        print()
    
    if not results_train:
        print("❌ No valid results found.")
        return
    
    # Create DataFrame and rank by IC
    df_train = pd.DataFrame(results_train)
    df_train = df_train.sort_values("IC", ascending=False)
    
    # Reorder columns
    cols = ["Factor_Name", "IC", "ICIR", "Ann_Alpha_%", "Alpha_Sharpe", "Turnover", "IC_Std", "NaN_IC_Rate_%", "N_Days", "Expression"]
    df_train = df_train[cols]
    
    print("\n" + "=" * 80)
    print("📊 TRAINING PERIOD RESULTS (2015-2022)")
    print("=" * 80)
    print(df_train.to_string(index=False))
    
    # Filter top performers
    top_factors = df_train[
        (df_train["IC"] > 0.03) & 
        (df_train["ICIR"] > 0.5) & 
        (df_train["Ann_Alpha_%"] > 2.0)
    ]
    
    print(f"\n🎯 Top Performers (IC > 0.03, ICIR > 0.5, Alpha > 2%):")
    print(f"   Found {len(top_factors)} factors\n")
    
    if len(top_factors) > 0:
        print(top_factors[["Factor_Name", "IC", "ICIR", "Ann_Alpha_%"]].to_string(index=False))
        
        # Validate on Out-of-Sample Period
        print(f"\n" + "=" * 80)
        print("🔍 OUT-OF-SAMPLE VALIDATION (2023-Present)")
        print("=" * 80)
        
        results_test = []
        for _, row in top_factors.iterrows():
            name = row["Factor_Name"]
            expr = row["Expression"]
            print(f"\n📊 Validating {name}...")
            res = evaluate_factor(expr, ETF_LIST, TEST_START, TEST_END, label_horizon=1)
            if res:
                res["Factor_Name"] = name
                results_test.append(res)
                print(f"   ✓ OOS IC={res['IC']:.4f}, ICIR={res['ICIR']:.2f}, Alpha={res['Ann_Alpha_%']:.2f}%")
        
        if results_test:
            df_test = pd.DataFrame(results_test)
            df_test = df_test.sort_values("IC", ascending=False)
            df_test = df_test[cols]
            
            print("\n" + "=" * 80)
            print("📈 OUT-OF-SAMPLE RESULTS")
            print("=" * 80)
            print(df_test.to_string(index=False))
            
            # Check correlation with existing factors
            print(f"\n" + "=" * 80)
            print("🔗 CORRELATION CHECK WITH EXISTING FACTORS")
            print("=" * 80)
            
            existing_exprs, existing_names = get_existing_factors()
            print(f"Existing factors: {', '.join(existing_names)}\n")
            
            for _, row in df_test.iterrows():
                name = row["Factor_Name"]
                expr = row["Expression"]
                print(f"📊 {name}:")
                corrs = calculate_factor_correlation(expr, existing_exprs, ETF_LIST, TRAIN_START, TRAIN_END)
                if corrs:
                    max_corr = max(abs(v) for v in corrs.values())
                    print(f"   Max Correlation: {max_corr:.3f}")
                    high_corr = {k: v for k, v in corrs.items() if abs(v) > 0.7}
                    if high_corr:
                        print(f"   ⚠️  High correlation (>0.7): {high_corr}")
                    else:
                        print(f"   ✓ Low correlation - Factor is unique!")
                print()
    
    # Save results
    output_file = "artifacts/alpha_mining_results.csv"
    df_train.to_csv(output_file, index=False)
    print(f"\n💾 Results saved to {output_file}")
    
    if len(top_factors) > 0:
        oos_file = "artifacts/alpha_mining_oos_validation.csv"
        if results_test:
            df_test.to_csv(oos_file, index=False)
            print(f"💾 OOS validation saved to {oos_file}")
    
    print("\n" + "=" * 80)
    print("✅ Alpha Mining Complete!")
    print("=" * 80)

if __name__ == "__main__":
    main()
