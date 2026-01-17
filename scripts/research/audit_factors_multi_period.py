
import qlib
import pandas as pd
import numpy as np
import os
import sys
from qlib.data import D

# Add workspace to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from delorean.conf import QLIB_PROVIDER_URI, QLIB_REGION, ETF_LIST
from delorean.data.handlers import ETFDataHandler

def audit_factors():
    qlib.init(provider_uri=QLIB_PROVIDER_URI, region=QLIB_REGION)
    
    # 1. Initialize Handler to get features and labels
    handler = ETFDataHandler(
        instruments=ETF_LIST,
        start_time="2018-01-01",
        end_time="2025-12-31",
        label_horizon=1
    )
    
    # Fetch data
    print("Fetching factor and label data...")
    df = handler.fetch()
    
    # Identify features and labels based on flat structure confirmed in research
    # All columns except the last one are features
    # The last one is the dynamic horizon return label
    col_names = df.columns.tolist()
    label_col = col_names[-1]
    feature_cols = col_names[:-1]
    
    print(f"Detected features: {feature_cols}")
    print(f"Detected label: {label_col}")
    
    # Ensure multi-index (datetime, instrument)
    df.index = df.index.set_names(['datetime', 'instrument'])
    
    features = df[feature_cols]
    label = df[label_col]
    
    periods = {
        "Bull Market (2019-2020)": ("2019-01-01", "2020-12-31"),
        "Transition (2021-2022)": ("2021-01-01", "2022-12-31"),
        "Bear/Choppy (2023-2024)": ("2023-01-01", "2024-12-31"),
        "Recent (2025)": ("2025-01-01", "2025-12-31")
    }
    
    results = []
    
    for period_name, (start, end) in periods.items():
        print(f"Auditing period: {period_name} ({start} to {end})...")
        
        # Slice data
        p_features = features.loc[start:end]
        p_label = label.loc[start:end]
        
        if p_features.empty:
            print(f"  > Warning: No data for {period_name}")
            continue
            
        for col in p_features.columns:
            # Calculate daily Rank IC
            combined = pd.DataFrame({"score": p_features[col], "label": p_label}).dropna()
            if combined.empty: continue
            
            daily_ic = combined.groupby(level='datetime').apply(
                lambda x: x["score"].corr(x["label"], method="spearman")
            )
            
            mean_ic = daily_ic.mean()
            icir = mean_ic / daily_ic.std() if daily_ic.std() != 0 else 0
            
            results.append({
                "Period": period_name,
                "Factor": col,
                "Rank IC": mean_ic,
                "ICIR": icir
            })
            
    # Convert to DataFrame
    audit_df = pd.DataFrame(results)
    
    # Pivot for comparison
    pivot_ic = audit_df.pivot(index="Factor", columns="Period", values="Rank IC")
    pivot_icir = audit_df.pivot(index="Factor", columns="Period", values="ICIR")
    
    print("\n" + "="*80)
    print(f"{'FACTOR PERFORMANCE (RANK IC) BY PERIOD':^80}")
    print("="*80)
    print(pivot_ic.round(4))
    
    print("\n" + "="*80)
    print(f"{'FACTOR ROBUSTNESS (ICIR) BY PERIOD':^80}")
    print("="*80)
    print(pivot_icir.round(4))
    
    # Save results
    pivot_ic.to_csv("scripts/research/factor_audit_periods.csv")
    print("\nDetailed audit saved to scripts/research/factor_audit_periods.csv")

if __name__ == "__main__":
    audit_factors()
