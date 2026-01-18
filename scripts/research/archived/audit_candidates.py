
import qlib
import pandas as pd
import numpy as np
import os
import sys
from qlib.data import D

# Add workspace to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from delorean.conf import QLIB_PROVIDER_URI, QLIB_REGION, ETF_LIST
from qlib.data.dataset.handler import DataHandlerLP

def audit_candidates():
    qlib.init(provider_uri=QLIB_PROVIDER_URI, region=QLIB_REGION)
    
    # 1. Define current + new candidates
    factors = [
        # EXISTING
        ("$close / Ref($close, 60) - 1", "MOM60"),
        ("$close / Ref($close, 120) - 1", "MOM120"),
        ("Mean(-1 * Corr($close / Ref($close, 1), $volume / Ref($volume, 1), 10), 5)", "Vol_Price_Div_Rev"),
        ("Mean(Sum((($close - $low) - ($high - $close)) / ($high - $low + 0.001) * $volume, 20) / Sum($volume, 20), 5)", "Money_Flow_20"),
        ("($close - Min($low, 20)) / (Max($high, 20) - Min($low, 20) + 1e-4)", "Range_Pos_20"),
        ("-1 * (Sum(-1 * (Log($open)), 5) + Std($high, 5))", "Alpha_Gen_8"),
        
        # CANDIDATES
        ("($close / Ref($close, 20) - 1) / (Std($close / Ref($close, 1) - 1, 20) + 1e-4)", "RiskAdjMom_20"),
        ("Corr($close / Ref($close, 1) - 1, $volume / Mean($volume, 20), 10)", "Vol_Confirm_10")
    ]
    
    exprs, names = list(zip(*factors))
    
    # 2. Setup DataHandler
    handler_config = {
        "start_time": "2018-01-01",
        "end_time": "2025-12-31",
        "instruments": ETF_LIST,
        "data_loader": {
            "class": "QlibDataLoader",
            "kwargs": {
                "config": {
                    "feature": (list(exprs), list(names)),
                    "label": ["Ref($close, -5) / $close - 1"] # 5-day return
                },
                "freq": "day",
            }
        },
        "learn_processors": [
             {"class": "DropnaLabel"},
             {"class": "CSZScoreNorm", "kwargs": {"fields_group": "feature"}} 
        ]
    }
    
    print("Fetching factor data for audit...")
    handler = DataHandlerLP(**handler_config)
    df = handler.fetch()
    df.index = df.index.set_names(['datetime', 'instrument'])
    
    col_names = df.columns.tolist()
    label_col = col_names[-1]
    feature_cols = col_names[:-1]
    
    features = df[feature_cols]
    label = df[label_col]
    
    # 3. Analyze Period: Recent (2023-2025)
    periods = {
        "Full (2018-2025)": ("2018-01-01", "2025-12-31"),
        "Out-of-Sample (2023-2025)": ("2023-01-01", "2025-12-31")
    }
    
    results = []
    for period_name, (start, end) in periods.items():
        print(f"Auditing period: {period_name}...")
        p_features = features.loc[start:end]
        p_label = label.loc[start:end]
        
        if p_features.empty: continue
            
        for col in p_features.columns:
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
            
    audit_df = pd.DataFrame(results)
    
    print("\n" + "="*80)
    print(f"{'CANDIDATE FACTOR PERFORMANCE':^80}")
    print("="*80)
    print(audit_df[audit_df["Period"] == "Out-of-Sample (2023-2025)"].sort_values("Rank IC", ascending=False))
    
    # Save results
    audit_df.to_csv("artifacts/candidate_audit_integrated.csv")

if __name__ == "__main__":
    audit_candidates()
