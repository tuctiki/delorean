#!/usr/bin/env python3
"""
Proper Turnover Validation
Calculate turnover correctly: average rank correlation between consecutive days.
High correlation (>0.4) = low turnover = stable signals
"""
import sys
import os
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import qlib
from qlib.data import D

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from delorean.config import QLIB_PROVIDER_URI, QLIB_REGION, ETF_LIST
from delorean.data import ETFDataHandler

def init_qlib():
    provider_uri = os.path.expanduser(QLIB_PROVIDER_URI)
    qlib.init(provider_uri=provider_uri, region=QLIB_REGION)
    print(f"✓ Qlib initialized: {provider_uri}\n")

def calculate_proper_turnover(factor_data):
    """
    Calculate turnover as 1 - rank_autocorrelation.
    Low turnover (<0.6) means high stability (rank_corr > 0.4)
    """
    try:
        # Get ranks for each day
        daily_ranks = factor_data.groupby(level='datetime').rank(pct=True)
        
        # Unstack to get time x ticker matrix
        ranks_df = daily_ranks.unstack()
        
        # Calculate rank correlation between consecutive days
        correlations = []
        for i in range(1, len(ranks_df)):
            prev_day = ranks_df.iloc[i-1].dropna()
            curr_day = ranks_df.iloc[i].dropna()
            
            # Find common tickers
            common = prev_day.index.intersection(curr_day.index)
            if len(common) >= 3:
                corr, _ = spearmanr(prev_day[common], curr_day[common])
                if not np.isnan(corr):
                    correlations.append(corr)
        
        if correlations:
            avg_rank_corr = np.mean(correlations)
            turnover = 1 - avg_rank_corr  # Turnover = instability
            return {
                'avg_rank_autocorr': avg_rank_corr,
                'turnover': turnover,
                'stable_days': len(correlations)
            }
        else:
            return None
    except Exception as e:
        print(f"Error calculating turnover: {e}")
        return None

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate Factor Turnover')
    parser.add_argument('--start', type=str, default="2023-01-01")
    parser.add_argument('--end', type=str, default="2025-12-31")
    args = parser.parse_args()
    
    print("=" * 80)
    print("🔄 PROPER TURNOVER VALIDATION")
    print("=" * 80)
    print("\nTurnover = 1 - Avg(Rank_Autocorrelation)")
    print("Good target: Turnover < 0.6 (Rank_Corr > 0.4)")
    print("=" * 80)
    
    init_qlib()
    
    # Get current factors
    exprs, names = ETFDataHandler.get_custom_factors()
    print(f"\n📊 Validating {len(names)} factors")
    print(f"📅 Period: {args.start} to {args.end}\n")
    
    # Load each factor
    results = []
    for expr, name in zip(exprs, names):
        print(f"📈 {name}...")
        try:
            df = D.features(ETF_LIST, [expr], start_time=args.start, end_time=args.end, freq='day')
            df.columns = [name]
            df = df.dropna()
            
            if len(df) < 100:
                print(f"   ⚠️  Insufficient data")
                continue
            
            metrics = calculate_proper_turnover(df[name])
            if metrics:
                print(f"   Rank Autocorr: {metrics['avg_rank_autocorr']:.3f}")
                print(f"   Turnover: {metrics['turnover']:.1%}")
                
                # Interpret
                if metrics['turnover'] < 0.4:
                    status = "✅ EXCELLENT"
                elif metrics['turnover'] < 0.6:
                    status = "✓ GOOD"
                elif metrics['turnover'] < 0.8:
                    status = "⚠️  HIGH"
                else:
                    status = "❌ EXTREME"
                
                print(f"   Status: {status}\n")
                
                results.append({
                    'Factor': name,
                    'Rank_Autocorr': metrics['avg_rank_autocorr'],
                    'Turnover': metrics['turnover'],
                    'Stable_Days': metrics['stable_days']
                })
        except Exception as e:
            print(f"   ❌ Error: {e}\n")
    
    if results:
        df_results = pd.DataFrame(results)
        df_results = df_results.sort_values('Turnover')
        
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(df_results.to_string(index=False))
        
        # Save
        output = "artifacts/turnover_validation.csv"
        df_results.to_csv(output, index=False)
        print(f"\n💾 Saved to {output}")
        
        # Overall assessment
        avg_turnover = df_results['Turnover'].mean()
        print(f"\n📊 Average Turnover: {avg_turnover:.1%}")
        if avg_turnover < 0.6:
            print("✅ PASS: Library has acceptable turnover")
        else:
            print("❌ FAIL: Library needs more smoothing")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
