
import sys
import os
import pandas as pd
import numpy as np
import qlib
from qlib.data import D

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from delorean.config import QLIB_PROVIDER_URI, QLIB_REGION, ETF_LIST
from delorean.data import ETFDataHandler

def init_qlib():
    provider_uri = os.path.expanduser(QLIB_PROVIDER_URI)
    qlib.init(provider_uri=provider_uri, region=QLIB_REGION)

def fetch_regime_signal(start, end):
    # Benchmark: CSI 300 ETF
    BENCHMARK = "510300.SH" 
    # Regime Signal: Price / MA60
    # Note: Qlib expression engine
    fields = ["$close / Mean($close, 60)"]
    df = D.features([BENCHMARK], fields, start_time=start, end_time=end, freq='day')
    df.columns = ["regime_ratio"]
    return df.droplevel('instrument')

def main():
    init_qlib()
    
    START = "2020-01-01"
    END = "2025-12-31"
    
    print(f"Loading Data ({START} to {END})...")
    
    # 1. Get Factors
    exprs, names = ETFDataHandler.get_custom_factors()
    
    # 2. Get Label
    label_expr = "Ref($close, -5) / $close - 1"
    
    fields = list(exprs) + [label_expr]
    cols = list(names) + ["label"]
    
    df = D.features(ETF_LIST, fields, start_time=START, end_time=END, freq='day')
    df.columns = cols
    
    # 3. Get Regime Signal
    regime = fetch_regime_signal(START, END)
    
    # Merge
    # Align dates
    common_dates = df.index.get_level_values('datetime').unique().intersection(regime.index.get_level_values('datetime').unique())
    
    # Calculate Daily IC per factor
    print("Calculating Daily IC...")
    daily_ics = {}
    
    for name in names:
        # Group by date, calculate Spearman corr with label
        ic_series = df.groupby("datetime").apply(lambda x: x[name].corr(x["label"], method="spearman"))
        daily_ics[name] = ic_series
        
    df_ic = pd.DataFrame(daily_ics)
    
    # Merge with Regime
    analysis_df = df_ic.join(regime, how='inner')
    
    print("\n" + "="*60)
    print("REGIME ANALYSIS RESULTS")
    print("="*60)
    print(f"Data Points: {len(analysis_df)}")
    
    # Analyze correlation between Regime Ratio and Factor IC
    # Positive Corr: Factor works better in Bull (High Ratio)
    # Negative Corr: Factor works better in Bear (Low Ratio)
    
    results = []
    
    for name in names:
        corr = analysis_df[name].corr(analysis_df["regime_ratio"])
        
        # Test "Flipped" Strategy
        # If Regime > 1.0 (Bull): Use Factor
        # If Regime < 1.0 (Bear): Use -Factor
        
        # Calculate IC of the conditional factor
        # We approximate this by: weighted_ic = mean( IC * sign(regime - 1) ) ? 
        # Actually easier: 
        #   Bull IC = Mean IC when regime > 1
        #   Bear IC = Mean IC when regime < 1
        
        bull_mask = analysis_df["regime_ratio"] > 1.0
        bear_mask = analysis_df["regime_ratio"] <= 1.0
        
        ic_bull = analysis_df.loc[bull_mask, name].mean()
        ic_bear = analysis_df.loc[bear_mask, name].mean()
        
        ic_flipped = (ic_bull * bull_mask.sum() - ic_bear * bear_mask.sum()) / (bull_mask.sum() + bear_mask.sum())
        
        original_ic = analysis_df[name].mean()
        
        status = "NEUTRAL"
        if ic_bull > 0 and ic_bear < 0:
            status = "FLIPPABLE (Bull+, Bear-)"
        elif ic_bull < 0 and ic_bear > 0:
            status = "FLIPPABLE (Bull-, Bear+)"
            
        results.append({
            "Factor": name,
            "IC_Corr_Regime": corr,
            "IC_Bull": ic_bull,
            "IC_Bear": ic_bear,
            "IC_Orig": original_ic,
            "IC_Flipped": ic_flipped if status.startswith("FLIPPABLE") else original_ic,
            "Status": status
        })
        
    res_df = pd.DataFrame(results)
    print(res_df.round(4).to_string(index=False))
    
    res_df.to_csv("artifacts/regime_analysis.csv", index=False)
    print("\nSaved to artifacts/regime_analysis.csv")

if __name__ == "__main__":
    main()
