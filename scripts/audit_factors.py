import sys
import os
import pandas as pd
import numpy as np
import qlib
from qlib.data import D
from qlib.config import REG_CN

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from delorean.config import QLIB_PROVIDER_URI, QLIB_REGION, ETF_LIST
from delorean.data import ETFDataHandler

def init_qlib():
    provider_uri = os.path.expanduser(QLIB_PROVIDER_URI)
    qlib.init(provider_uri=provider_uri, region=QLIB_REGION)

def calculate_metrics(df_factors, df_label):
    """
    Calculate metrics for all factors at once.
    df_factors: DataFrame (datetime, instrument) -> columns=factor_names
    df_label: Series (datetime, instrument) -> label
    """
    results = []
    
    # 1. IC and ICIR
    # Align indices
    common_idx = df_factors.index.intersection(df_label.index)
    df_f = df_factors.loc[common_idx]
    label = df_label.loc[common_idx]
    
    # Rank IC
    # Group by datetime
    def rank_ic(group):
        if group.empty: return pd.Series(index=df_f.columns)
        # Rank the label for this day
        # Actually corr(method='spearman') does rank automatically
        # But we need to handle multiple columns against one label
        l = label.loc[group.index]
        return group.apply(lambda col: col.corr(l, method='spearman'))

    daily_ic = df_f.groupby("datetime").apply(rank_ic)
    
    mean_ic = daily_ic.mean()
    std_ic = daily_ic.std()
    icir = mean_ic / std_ic
    
    # 2. Turnover
    # Daily turnover of the RANKED signal (normalized 0-1)
    # This simulates portfolio turnover if we traded 100% based on this factor
    def get_rank_turnover(series):
        # Rank per day
        ranks = series.groupby("datetime").rank(pct=True)
        # Delta
        # We need to preserve instrument alignment
        # Unstack to (datetime, instrument)
        ranks_unstacked = ranks.unstack()
        delta = ranks_unstacked.diff().abs().sum(axis=1) # Sum of absolute changes across all assets
        # Normalize by number of assets? Or just total churn?
        # Usually turnover = sum(abs(delta)) / 2 (since buy+sell)
        # Percentage of portfolio replaced.
        return (delta / 2).mean() # Mean daily turnover

    # 3. Correlation Matrix
    corr_matrix = df_f.corr(method='pearson')
    
    # Compile results
    for name in df_factors.columns:
        ic = mean_ic[name]
        ir = icir[name]
        
        # Turnover calculation (expensive, so maybe simplify or do per factor)
        # For efficiency, we calculate turnover outside loop if possible, but per-col is fine
        turnover = get_rank_turnover(df_factors[name])
        
        # Max Correlation with OTHER factors
        # Drop self
        others = corr_matrix[name].drop(name)
        if not others.empty:
            max_corr = others.max()
            max_corr_partner = others.idxmax()
        else:
            max_corr = 0.0
            max_corr_partner = "None"
            
        # Status Triage
        status = "KEEP"
        reason = []
        
        if ic < 0.01 and ic > -0.01:
            status = "REMOVE"
            reason.append("Low Signal")
        elif abs(ic) < 0.03:
            status = "REWORK"
            reason.append("Weak Signal")
            
        if turnover > 0.6: # > 60% turnover
            if status == "KEEP": status = "REWORK"
            reason.append("High Turnover")
            
        if max_corr > 0.7:
            # Check if we are the weaker one?
            # For now just flag collision
            reason.append(f"High Corr with {max_corr_partner}")
            if status == "KEEP": status = "CHECK_REDUNDANCY"

        results.append({
            "Name": name,
            "IC": ic,
            "ICIR": ir,
            "Turnover": turnover,
            "Max_Corr": max_corr,
            "Corr_With": max_corr_partner,
            "Status": status,
            "Reason": "; ".join(reason)
        })
        
    return pd.DataFrame(results), corr_matrix

def main():
    print("Initializing Qlib...")
    init_qlib()
    
    # 1. Get Factors
    exprs, names = ETFDataHandler.get_custom_factors()
    print(f"Auditing {len(exprs)} factors...")
    
    # 2. Load Data (Recent Period)
    # Current time 2026. Audit 2024-2025.
    START = "2024-01-01"
    END = "2025-12-31"
    print(f"Data Period: {START} to {END}")
    
    fields = exprs + ["Ref($close, -1) / $close - 1"] # Add Label
    field_names = names + ["label"]
    
    # Fetch all at once
    df = D.features(ETF_LIST, fields, start_time=START, end_time=END, freq='day')
    df.columns = field_names
    df = df.dropna()
    
    if df.empty:
        print("No data found for audit period.")
        return
        
    df_factors = df[names]
    df_label = df["label"]
    
    print("Calculating metrics (IC, Turnover, Correlation)...")
    results, corr_matrix = calculate_metrics(df_factors, df_label)
    
    # Sort by Status priority then IC
    status_order = {"KEEP": 0, "CHECK_REDUNDANCY": 1, "REWORK": 2, "REMOVE": 3}
    results["SortKey"] = results["Status"].map(status_order)
    results = results.sort_values(["SortKey", "IC"], ascending=[True, False]).drop("SortKey", axis=1)
    
    print("\n=== Factor Audit Report ===")
    print(results.to_string(index=False, float_format=lambda x: "{:.4f}".format(x)))
    
    results.to_csv("artifacts/factor_audit.csv", index=False)
    print("\nSaved audit to artifacts/factor_audit.csv")

    # Suggest High Level Actions
    print("\n=== Recommendations ===")
    for _, row in results.iterrows():
        if row['Status'] != 'KEEP':
            print(f"- {row['Name']}: {row['Status']} ({row['Reason']})")

if __name__ == "__main__":
    main()
