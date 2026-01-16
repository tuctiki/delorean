
import qlib
from qlib.data import D
from qlib.contrib.evaluate import risk_analysis
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import logging
from delorean.config import (
    QLIB_PROVIDER_URI, QLIB_REGION, ETF_LIST, START_TIME, END_TIME
)
from delorean.data import ETFDataLoader

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("audit_factors")

def analyze_factors():
    # 1. Initialize Qlib
    qlib.init(provider_uri=QLIB_PROVIDER_URI, region=QLIB_REGION)
    
    # 2. Load Data (2024-2025 for "Recent" Audit)
    # Using the last 2 years for relevancy
    audit_start = "2024-01-01"
    audit_end = "2025-12-31" 
    
    logger.info(f"Starting Factor Audit for period: {audit_start} to {audit_end}")
    
    loader = ETFDataLoader(label_horizon=1)
    
    # Load dataset explicitly to get the feature columns
    # We want to inspect the individual factors, not just the model prediction
    # So we need to fetch the raw factors again or use the cached dataset logic if accessible
    
    # Simpler approach: Re-fetch factors using definition in data.py
    from delorean.data import ETFDataHandler
    exprs, names = ETFDataHandler.get_custom_factors()
    
    if not exprs:
        logger.error("No custom factors found!")
        return

    logger.info(f"Auditing {len(exprs)} Factors: {names}")
    
    # Fetch data
    try:
        # Fetch features + label
        # Label: Ref($close, -1) / $close - 1
        fields = exprs + ["Ref($close, -1) / $close - 1"]
        col_names = names + ["label"]
        
        df = D.features(ETF_LIST, fields, start_time=audit_start, end_time=audit_end)
        df.columns = col_names
        
    except Exception as e:
        logger.error(f"Failed to fetch data: {e}")
        return

    # 3. Calculate Metrics (IC, RankIC)
    results = []
    
    for factor in names:
        # Drop NaNs
        sub_df = df[[factor, "label"]].dropna()
        
        if sub_df.empty:
            logger.warning(f"Factor {factor} has no valid data.")
            continue
            
        # IC: Pearson Correlation
        ic, _ = pearsonr(sub_df[factor], sub_df["label"])
        
        # RankIC: Spearman Correlation
        rank_ic = sub_df[[factor, "label"]].corr(method="spearman").iloc[0, 1]
        
        results.append({
            "Factor": factor,
            "IC": ic,
            "RankIC": rank_ic
        })
        
    audit_df = pd.DataFrame(results)
    print("\n=== FACTOR PERFORMANCE AUDIT (2024-2025) ===")
    print(audit_df.sort_values("RankIC", ascending=False))
    
    # 4. Correlation Matrix
    print("\n=== FACTOR CORRELATION MATRIX ===")
    factor_data = df[names].dropna()
    corr_matrix = factor_data.corr()
    print(corr_matrix)
    
    # 5. Recommendations (Simple Rules)
    print("\n=== RECOMMENDATIONS ===")
    for _, row in audit_df.iterrows():
        status = "KEEP"
        reasons = []
        
        if abs(row['RankIC']) < 0.03:
            status = "REMOVE/REWORK"
            reasons.append("Low Predictive Power (RankIC < 0.03)")
            
        print(f"Factor: {row['Factor']:<20} | Status: {status:<15} | IC: {row['IC']:.4f} | RankIC: {row['RankIC']:.4f} | Note: {', '.join(reasons)}")

if __name__ == "__main__":
    analyze_factors()
