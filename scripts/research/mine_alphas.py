
import qlib
from qlib.data import D
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import logging
from delorean.config import (
    QLIB_PROVIDER_URI, QLIB_REGION, ETF_LIST
)
# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mine_alphas")

def mine_alphas():
    # 1. Initialize Qlib
    qlib.init(provider_uri=QLIB_PROVIDER_URI, region=QLIB_REGION)
    
    # 2. Config
    audit_start = "2024-01-01"
    audit_end = "2025-12-31" 
    
    # 3. Candidate Pool (Phase 15: Trend Efficiency & Volume Flow)
    # Focus: 5-Day Horizon, Smoothed for Stability
    candidates = {
        # Efficiency Ratio (Kaufman): Abs(Net Change) / Sum(Abs(Change))
        # Measures trend smoothness. Higher = Smoother trend.
        "Efficiency_Ratio_20": "Mean(Abs($close - Ref($close, 20)) / Sum(Abs($close - Ref($close, 1)), 20), 5)",
        
        # Money Flow Multiplier Proxy:
        # CMF-like: Volume weighted by location in daily range
        "Money_Flow_20": "Mean(Sum((($close - $low) - ($high - $close)) / ($high - $low + 0.001) * $volume, 20) / Sum($volume, 20), 5)",
        
        # Volatility Breakout (Normalized):
        # Distance from 60d High, scaled by ATR
        "Vol_Breakout_60": "Mean(($close - Mean($close, 60)) / (Std($close, 60) + 0.001), 5)",
        
        # Volume Price Trend (VPT) - Simplified Rate of Change
        # Volume * %Price Change
        "VPT_Proxy_10": "Mean(Sum($volume * ($close/Ref($close, 1)-1), 10) / Mean($volume, 20), 5)",
        
        # Downside Deviation Ratio (Sortino-like proxy):
        # Upside Vol / Downside Vol
        "Up_Down_Vol_Ratio": "Mean(Std(If($close > Ref($close,1), $close/Ref($close,1)-1, 0), 20) / (Std(If($close < Ref($close,1), $close/Ref($close,1)-1, 0), 20) + 0.001), 5)",
    }
    
    logger.info(f"Mining {len(candidates)} Candidates on {audit_start} to {audit_end}...")
    
    # 4. Fetch Data
    # Align w/ Strategy: Use 5-day forward return as label
    fields = list(candidates.values()) + ["Ref($close, -5) / $close - 1"] # Label (5-Day)
    names = list(candidates.keys()) + ["label"]
    
    try:
        df = D.features(ETF_LIST, fields, start_time=audit_start, end_time=audit_end)
        df.columns = names
    except Exception as e:
        logger.error(f"Failed to fetch data: {e}")
        return

    # 5. Evaluate
    results = []
    
    for name in candidates.keys():
        sub_df = df[[name, "label"]].dropna()
        if sub_df.empty:
            continue
            
        # Metrics
        ic, _ = pearsonr(sub_df[name], sub_df["label"])
        rank_ic = sub_df[[name, "label"]].corr(method="spearman").iloc[0, 1]
        
        results.append({
            "Factor": name,
            "Formula": candidates[name],
            "IC": ic,
            "RankIC": rank_ic
        })
        
    # 6. Report
    audit_df = pd.DataFrame(results).sort_values("RankIC", ascending=False)
    print("\n=== ALPHA MINING RESULTS (2024-2025) ===")
    print(audit_df)
    
    # Correlation Check against existing best (Vol_Price_Div)
    # We need to fetch Vol_Price_Div separately or re-add it above. 
    # For now ensuring internal correlation is low
    print("\n=== INTERNAL CORRELATION MATRIX ===")
    print(df[list(candidates.keys())].corr())

if __name__ == "__main__":
    mine_alphas()
