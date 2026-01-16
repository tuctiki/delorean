
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
    
    # 3. Candidate Pool (Phase 12: Bull Trend Alphas - Reduced Turnover)
    # Focus: Longer lookbacks (30-60 days), Trend Confirmation, Stable signals
    candidates = {
        # Long-Term Momentum (60-day): More stable than 10-day
        "LT_Momentum_60": "($close / Ref($close, 60) - 1)",
        
        # Trend Strength (Price > MA): Positive only when in uptrend
        "Trend_Strength_30": "If($close > Mean($close, 30), ($close / Mean($close, 30) - 1), 0)",
        
        # Smooth Momentum (EMA-based): Less noisy than simple returns
        "Smooth_Momentum": "Mean($close / Ref($close, 1) - 1, 20)",
        
        # Price Position: Where is price in its 60-day range (0-1)
        "Price_Position": "($close - Min($close, 60)) / (Max($close, 60) - Min($close, 60) + 0.001)",
        
        # 52-Week High Proximity: Near high = strong trend
        "High_Proximity": "$close / Max($close, 60)",
        
        # Dual MA Cross: Bullish when fast > slow (binary -> continuous strength)
        "MA_Cross_Strength": "Mean($close, 10) / Mean($close, 30) - 1",
    }
    
    logger.info(f"Mining {len(candidates)} Candidates on {audit_start} to {audit_end}...")
    
    # 4. Fetch Data
    fields = list(candidates.values()) + ["Ref($close, -1) / $close - 1"] # Label
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
