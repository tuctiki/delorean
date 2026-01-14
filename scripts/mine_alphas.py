import sys
import os
import pandas as pd
import numpy as np
import qlib
from qlib.data import D
from qlib.data.dataset.loader import QlibDataLoader
from qlib.config import REG_CN

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from delorean.config import QLIB_PROVIDER_URI, QLIB_REGION, ETF_LIST, START_TIME, END_TIME

def init_qlib():
    provider_uri = os.path.expanduser(QLIB_PROVIDER_URI)
    qlib.init(provider_uri=provider_uri, region=QLIB_REGION)

def evaluate_factor(expression, instruments, start_time, end_time):
    """
    Evaluates a single factor expression.
    Returns: IC, IR, Long-Short Return
    """
    # 1. Load Factor and Label
    # Label: 1-day forward return
    fields = [expression, "Ref($close, -1) / $close - 1"]
    names = ["factor", "label"]
    
    try:
        data = D.features(
            instruments, 
            fields, 
            start_time=start_time, 
            end_time=end_time, 
            freq='day'
        )
        data.columns = names
        data = data.dropna()
        
        if data.empty:
            return None

        # 2. Calculate IC (Information Coefficient)
        # Group by date to calculate Rank IC per day
        # Factor rank vs Label rank
        daily_ic = data.groupby("datetime").apply(lambda df: df["factor"].corr(df["label"], method="spearman"))
        nan_ic_rate = daily_ic.isna().mean()
        daily_ic = daily_ic.dropna()
        
        mean_ic = daily_ic.mean()
        icir = mean_ic / daily_ic.std() if daily_ic.std() != 0 else 0
        
        # 3. Simple Backtest (Long Top 20%, Short Bottom 20% - Theoretical)
        # This is just a proxy for signal strength
        # We need per-day ranking
        def get_long_short_ret(df):
            try:
                # Top 20%
                top_cutoff = df["factor"].quantile(0.8)
                bot_cutoff = df["factor"].quantile(0.2)
                
                long_ret = df[df["factor"] >= top_cutoff]["label"].mean()
                short_ret = df[df["factor"] <= bot_cutoff]["label"].mean()
                
                # If shorting is not allowed (ETFs), we focus on Long Only return vs Mean
                # But for factor purity we usually look at spread.
                # Let's return Long - Market Mean
                market_ret = df["label"].mean()
                return long_ret - market_ret
            except:
                return 0.0

        daily_alpha = data.groupby("datetime").apply(get_long_short_ret)
        annualized_alpha = daily_alpha.mean() * 252
        
        return {
            "Expression": expression,
            "IC": mean_ic,
            "ICIR": icir,
            "Ann_Alpha_Ret": annualized_alpha,
            "NaN_IC_Rate": nan_ic_rate
        }

    except Exception as e:
        print(f"Error evaluating {expression}: {e}")
        return None

def main():
    init_qlib()
    
    # 2015-2022 for Training/Mining
    TRAIN_START = "2015-01-01"
    TRAIN_END = "2022-12-31"
    
    # Hypotheses to test
    candidates = [
        # 1. Volatility-Adjusted Momentum (20 days)
        # Rationale: Momentum is more reliable if volatiltiy is low.
        "( ($close / Ref($close, 20) - 1) / Std($close, 20) )",
        
        # 2. Mean Reversion on High Volume
        # Rationale: High volume change often signals capitulation or exhaustion.
        # If Price Down AND Volume High -> Reversal (Positive Return)
        # Formulated as: -1 * DeltaPrice * VolumeRatio
        # If Price drops (Delta negative) -> Factor positive.
        "(-1 * ($close - Ref($close, 1)) * ($volume / Mean($volume, 20)))",
        
        # 3. Acceleration (2nd Derivative)
        # Rationale: Rate of change of momentum.
        "($close - 2*Ref($close, 5) + Ref($close, 10))",
        
        # 4. ROC Ratio (Short vs Long term momentum)
        # Rationale: if short term momentum > long term, trend is accelerating.
        "( ($close/Ref($close, 5)) / ($close/Ref($close, 20)) )",
        
        # 5. Downside Deviatne Ratio
        # Rationale: Penalize downside volatility more? 
        # Simpler: Correlation between price and volume
        "Corr($close, $volume, 20)",
        
        # 6. Low Volatility Anomaly (Inverted Vol)
        # Rationale: Low vol stocks tend to outperform risk-adjusted.
        "(-1 * Std($close, 20))"
    ]
    
    results = []
    print(f"Mining {len(candidates)} factors on {len(ETF_LIST)} ETFs from {TRAIN_START} to {TRAIN_END}...")
    
    for expr in candidates:
        print(f"Testing: {expr}")
        res = evaluate_factor(expr, ETF_LIST, TRAIN_START, TRAIN_END)
        if res:
            results.append(res)
            
    df_res = pd.DataFrame(results)
    if not df_res.empty:
        df_res = df_res.sort_values("IC", ascending=False)
        print("\n=== Mining Results (Ranked by IC) ===")
        print(df_res.to_string(index=False))
        
        # Save to file
        df_res.to_csv("artifacts/alpha_mining_results.csv", index=False)
        print("\nSaved results to artifacts/alpha_mining_results.csv")
    else:
        print("No valid results found.")

if __name__ == "__main__":
    main()
