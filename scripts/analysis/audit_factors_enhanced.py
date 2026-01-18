#!/usr/bin/env python3
"""
Enhanced Alpha Factor Audit
Comprehensive evaluation of all 11 factors with detailed recommendations.
"""
import sys
import os
import pandas as pd
import numpy as np
import qlib
from qlib.data import D
from scipy.stats import spearmanr

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from delorean.config import QLIB_PROVIDER_URI, QLIB_REGION, ETF_LIST
from delorean.data import ETFDataHandler

def init_qlib():
    provider_uri = os.path.expanduser(QLIB_PROVIDER_URI)
    qlib.init(provider_uri=provider_uri, region=QLIB_REGION)
    print(f"✓ Qlib initialized: {provider_uri}\n")

def evaluate_single_factor(factor_data, label_data, factor_name):
    """Evaluate a single factor's performance."""
    try:
        # Align indices
        common_idx = factor_data.index.intersection(label_data.index)
        factor = factor_data.loc[common_idx]
        label = label_data.loc[common_idx]
        
        if len(factor) < 50:
            return None
        
        # Calculate daily IC (Spearman correlation)
        daily_ic = factor.groupby("datetime").apply(
            lambda group: group.corr(label.loc[group.index], method="spearman")
        )
        daily_ic = daily_ic.dropna()
        
        if len(daily_ic) < 10:
            return None
        
        mean_ic = daily_ic.mean()
        ic_std = daily_ic.std()
        icir = mean_ic / ic_std if ic_std != 0 else 0
        
        # Calculate turnover (rank change)
        def calc_turnover():
            try:
                ranks = factor.groupby("datetime").rank(pct=True)
                ranks_unstacked = ranks.unstack(fill_value=0)
                delta = ranks_unstacked.diff().abs().sum(axis=1)
                return (delta / 2).mean()
            except:
                return np.nan
        
        turnover = calc_turnover()
        
        # Calculate long-only alpha
        def get_long_alpha(group):
            try:
                top_cutoff = group.quantile(0.8)
                long_ret = label.loc[group[group >= top_cutoff].index].mean()
                market_ret = label.loc[group.index].mean()
                return long_ret - market_ret
            except:
                return np.nan
        
        daily_alpha = factor.groupby("datetime").apply(get_long_alpha).dropna()
        ann_alpha = daily_alpha.mean() * 252
        alpha_sharpe = (daily_alpha.mean() / daily_alpha.std() * np.sqrt(252)) if daily_alpha.std() != 0 else 0
        
        return {
            "IC": mean_ic,
            "IC_Std": ic_std,
            "ICIR": icir,
            "Turnover": turnover,
            "Ann_Alpha_%": ann_alpha * 100,
            "Alpha_Sharpe": alpha_sharpe,
            "N_Days": len(daily_ic)
        }
    except Exception as e:
        print(f"  ⚠️  Error evaluating {factor_name}: {e}")
        return None

def calculate_correlation_matrix(df_factors):
    """Calculate correlation matrix between all factors."""
    return df_factors.corr(method='pearson')

def triage_factor(metrics, corr_row, factor_name):
    """Classify factor as KEEP, REMOVE, or REWORK."""
    status = "KEEP"
    reasons = []
    recommendations = []
    
    ic = metrics["IC"]
    icir = metrics["ICIR"]
    turnover = metrics["Turnover"]
    
    # IC-based triage
    if abs(ic) < 0.01:
        status = "REMOVE"
        reasons.append("Very Low Signal (|IC| < 0.01)")
        recommendations.append("Consider removing - no predictive power")
    elif abs(ic) < 0.02:
        status = "REWORK"
        reasons.append("Weak Signal (|IC| < 0.02)")
        recommendations.append("Try different lookback windows or add filters")
    
    # ICIR consistency check
    if abs(icir) < 0.3 and status == "KEEP":
        status = "REWORK"
        reasons.append("Inconsistent (|ICIR| < 0.3)")
        recommendations.append("Signal is noisy - consider smoothing or regime filters")
    
    # Turnover check
    if not np.isnan(turnover) and turnover > 0.6:
        if status == "KEEP":
            status = "REWORK"
        reasons.append(f"High Turnover ({turnover:.1%})")
        recommendations.append("Apply EWMA smoothing or increase lookback window")
    
    # Correlation check
    corr_others = corr_row.drop(factor_name)
    if not corr_others.empty:
        max_corr = corr_others.abs().max()
        max_corr_partner = corr_others.abs().idxmax()
        
        if max_corr > 0.7:
            if status == "KEEP":
                status = "CHECK_REDUNDANCY"
            reasons.append(f"High Correlation with {max_corr_partner} ({max_corr:.2f})")
            recommendations.append(f"Consider removing if {max_corr_partner} performs better")
    else:
        max_corr = 0
        max_corr_partner = "None"
    
    return {
        "Status": status,
        "Reasons": "; ".join(reasons) if reasons else "Good performance",
        "Recommendations": " | ".join(recommendations) if recommendations else "Keep as is",
        "Max_Corr": max_corr,
        "Corr_With": max_corr_partner
    }

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Audit Alpha Factors')
    parser.add_argument('--start', type=str, default="2023-01-01", help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default="2025-12-31", help='End date (YYYY-MM-DD)')
    parser.add_argument('--output', type=str, default="artifacts/factor_audit_detailed.csv", help='Output CSV path')
    args = parser.parse_args()
    
    print("=" * 80)
    print("🔍 ALPHA FACTOR AUDIT - Comprehensive Evaluation")
    print("=" * 80)
    
    init_qlib()
    
    # Get current factor library
    exprs, names = ETFDataHandler.get_custom_factors()
    print(f"📊 Auditing {len(names)} factors:")
    for i, name in enumerate(names, 1):
        print(f"  {i}. {name}")
    
    # Audit period
    START = args.start
    END = args.end
    LABEL_HORIZON = 5
    
    print(f"\n📅 Audit Period: {START} to {END}")
    print(f"🎯 Label: {LABEL_HORIZON}-day forward return\n")
    
    # Load all factors and label
    label_expr = f"Ref($close, -{LABEL_HORIZON}) / $close - 1"
    fields = exprs + [label_expr]
    field_names = names + ["label"]
    
    print("📥 Loading data...")
    df = D.features(ETF_LIST, fields, start_time=START, end_time=END, freq='day')
    df.columns = field_names
    df = df.dropna()
    
    if df.empty:
        print("❌ No data found for audit period.")
        return
    
    print(f"✓ Loaded {len(df)} data points\n")
    
    df_factors = df[names]
    df_label = df["label"]
    
    # Evaluate each factor
    print("=" * 80)
    print("INDIVIDUAL FACTOR EVALUATION")
    print("=" * 80)
    
    results = []
    for name in names:
        print(f"\n📈 Evaluating: {name}")
        metrics = evaluate_single_factor(df_factors[name], df_label, name)
        
        if metrics:
            print(f"   IC: {metrics['IC']:.4f}, ICIR: {metrics['ICIR']:.2f}, "
                  f"Alpha: {metrics['Ann_Alpha_%']:.1f}%, Turnover: {metrics['Turnover']:.1%}")
            results.append({"Name": name, **metrics})
        else:
            print(f"   ⚠️  Insufficient data")
    
    if not results:
        print("\n❌ No valid results.")
        return
    
    df_results = pd.DataFrame(results)
    
    # Calculate correlation matrix
    print(f"\n{'=' * 80}")
    print("CORRELATION ANALYSIS")
    print("=" * 80)
    
    corr_matrix = calculate_correlation_matrix(df_factors)
    print("\nCorrelation Matrix:")
    print(corr_matrix.round(2))
    
    # Triage each factor
    print(f"\n{'=' * 80}")
    print("FACTOR TRIAGE & RECOMMENDATIONS")
    print("=" * 80)
    
    triage_results = []
    for _, row in df_results.iterrows():
        name = row["Name"]
        triage = triage_factor(row.to_dict(), corr_matrix[name], name)
        triage_results.append({"Name": name, **triage})
    
    df_triage = pd.DataFrame(triage_results)
    
    # Merge results
    df_final = df_results.merge(df_triage, on="Name")
    
    # Sort by status priority
    status_order = {"KEEP": 0, "CHECK_REDUNDANCY": 1, "REWORK": 2, "REMOVE": 3}
    df_final["SortKey"] = df_final["Status"].map(status_order)
    df_final = df_final.sort_values(["SortKey", "IC"], ascending=[True, False]).drop("SortKey", axis=1)
    
    # Display results
    print("\n" + "=" * 80)
    print("AUDIT SUMMARY")
    print("=" * 80)
    
    display_cols = ["Name", "IC", "ICIR", "Ann_Alpha_%", "Alpha_Sharpe", "Turnover", 
                    "Max_Corr", "Corr_With", "Status"]
    print(df_final[display_cols].to_string(index=False))
    
    # Status breakdown
    print(f"\n{'=' * 80}")
    print("STATUS BREAKDOWN")
    print("=" * 80)
    
    status_counts = df_final["Status"].value_counts()
    for status, count in status_counts.items():
        print(f"{status}: {count} factors")
    
    # Detailed recommendations
    print(f"\n{'=' * 80}")
    print("DETAILED RECOMMENDATIONS")
    print("=" * 80)
    
    for _, row in df_final.iterrows():
        if row["Status"] != "KEEP":
            print(f"\n📌 {row['Name']}")
            print(f"   Status: {row['Status']}")
            print(f"   Reasons: {row['Reasons']}")
            print(f"   Recommendations: {row['Recommendations']}")
    
    # Identify redundancy clusters
    print(f"\n{'=' * 80}")
    print("REDUNDANCY ANALYSIS")
    print("=" * 80)
    
    high_corr_pairs = []
    for i, name1 in enumerate(names):
        for name2 in names[i+1:]:
            corr = corr_matrix.loc[name1, name2]
            if abs(corr) > 0.7:
                ic1 = df_final[df_final["Name"] == name1]["IC"].values[0]
                ic2 = df_final[df_final["Name"] == name2]["IC"].values[0]
                high_corr_pairs.append({
                    "Factor 1": name1,
                    "Factor 2": name2,
                    "Correlation": corr,
                    "IC1": ic1,
                    "IC2": ic2,
                    "Recommendation": f"Keep {name1 if abs(ic1) > abs(ic2) else name2}"
                })
    
    if high_corr_pairs:
        df_redundancy = pd.DataFrame(high_corr_pairs)
        print("\nHighly Correlated Pairs (|Corr| > 0.7):")
        print(df_redundancy.to_string(index=False))
    else:
        print("\n✓ No highly correlated pairs found (all |Corr| < 0.7)")
    
    # Save results
    output_file = args.output
    df_final.to_csv(output_file, index=False)
    print(f"\n💾 Detailed audit saved to {output_file}")
    
    corr_file = output_file.replace(".csv", "_corr.csv")
    corr_matrix.to_csv(corr_file)
    print(f"💾 Correlation matrix saved to {corr_file}")
    
    print(f"\n{'=' * 80}")
    print("✅ AUDIT COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
