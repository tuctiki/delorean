"""
Test Defensive Factors for ETF Strategy
Evaluates mean-reversion and volatility-based factors as alternatives to pure momentum.
"""
import qlib
from qlib.data import D
from delorean.config import QLIB_PROVIDER_URI, QLIB_REGION, ETF_LIST
import pandas as pd
import numpy as np

qlib.init(provider_uri=QLIB_PROVIDER_URI, region=QLIB_REGION)

# Defensive Factor Candidates
defensive_factors = {
    # RSI (14-day) - Mean Reversion Indicator
    # RSI = 100 - (100 / (1 + RS)), where RS = Avg Gain / Avg Loss
    # Simplified approximation using Qlib operations:
    "RSI_14": "(Mean(If($close > Ref($close, 1), $close - Ref($close, 1), 0), 14) / (Mean(Abs($close - Ref($close, 1)), 14) + 0.0001)) * 100 - 50",
    
    # Bollinger Band Position - Where is price in the channel?
    # (Price - Lower Band) / (Upper Band - Lower Band)
    # BB = MA(20) ± 2*STD(20)
    "BB_Position": "($close - (Mean($close, 20) - 2 * Std($close, 20))) / (4 * Std($close, 20) + 0.0001)",
    
    # ATR Percentile - Current volatility vs historical distribution
    # Inverted so low vol = high score (defensive)
    "ATR_Percentile_Inv": "-1 * (Mean(Max($high - $low, Abs($high - Ref($close, 1)), Abs($low - Ref($close, 1))), 14) / Mean($close, 14))",
    
    # RSI Divergence - RSI trend vs Price trend
    # Positive divergence = RSI rising while price falling (bullish)
    "RSI_Divergence": "Corr($close / Ref($close, 1) - 1, (Mean(If($close > Ref($close, 1), $close - Ref($close, 1), 0), 14) / (Mean(Abs($close - Ref($close, 1)), 14) + 0.0001)), 10)"
}

# Test Period
START = "2023-01-01"
END = "2025-12-31"

# Forward Return Label (5-day)
label_expr = "Ref($close, -5) / $close - 1"

if __name__ == '__main__':
    print("="*80)
    print("DEFENSIVE FACTOR TEST")
    print("="*80)
    print(f"Period: {START} to {END}")
    print(f"Assets: {len(ETF_LIST)} ETFs")
    print(f"Factors: {len(defensive_factors)}")
    print()
    
    results = []
    
    for name, expr in defensive_factors.items():
        print(f"Testing: {name}")
        try:
            # Load factor and label
            df = D.features(
                ETF_LIST,
                [expr, label_expr],
                start_time=START,
                end_time=END
            )
            df.columns = ['factor', 'label']
            df = df.dropna()
            
            if len(df) < 100:
                print(f"  ⚠️  Insufficient data ({len(df)} rows)")
                continue
            
            # Calculate IC (cross-sectional correlation)
            ic_series = df.groupby(level='datetime').apply(
                lambda x: x['factor'].corr(x['label'], method='spearman')
            )
            
            mean_ic = ic_series.mean()
            std_ic = ic_series.std()
            icir = mean_ic / (std_ic + 0.0001)
            
            # Stats
            result = {
                'Factor': name,
                'IC': mean_ic,
                'ICIR': icir,
                'IC_Std': std_ic,
                'Data_Points': len(df)
            }
            results.append(result)
            
            status = "✅ Strong" if abs(mean_ic) > 0.03 else "⚠️  Weak" if abs(mean_ic) > 0.01 else "❌ Noise"
            print(f"  IC: {mean_ic:.4f}, ICIR: {icir:.4f} {status}")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    print()
    print("="*80)
    print("SUMMARY")
    print("="*80)
    
    if results:
        df_results = pd.DataFrame(results).sort_values('IC', ascending=False, key=abs)
        print(df_results.to_string(index=False))
        
        print()
        print("RECOMMENDATION:")
        strong = df_results[df_results['IC'].abs() > 0.03]
        if len(strong) > 0:
            print(f"✅ Add {len(strong)} factors with |IC| > 0.03:")
            for _, row in strong.iterrows():
                print(f"   - {row['Factor']}: IC={row['IC']:.4f}")
        else:
            print("❌ No factors meet threshold (|IC| > 0.03). Consider mining alternatives.")
    else:
        print("No valid results.")
    
    print()
    print("="*80)
