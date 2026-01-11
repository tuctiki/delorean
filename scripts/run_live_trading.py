import qlib
import pandas as pd
import datetime
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qlib.workflow import R
from delorean.config import QLIB_PROVIDER_URI, QLIB_REGION, BENCHMARK
from delorean.data import ETFDataLoader
from delorean.model import ModelTrainer

def get_trading_signal(topk=4):
    """
    Generates trading signals for the latest available date.
    
    Args:
        topk (int): Number of top ETFs to select.
    """
    # 1. Initialize Qlib
    qlib.init(provider_uri=QLIB_PROVIDER_URI, region=QLIB_REGION)
    
    print("\n" + "="*50)
    print(f"  ETF Strategy Live Signal Generator")
    print("="*50)

    # 2. Load Data (All available history)
    print("[1/5] Loading Data...")
    data_loader = ETFDataLoader()
    # We load standard dataset. The time range is controlled by constants.py (set to 2099)
    dataset = data_loader.load_data()
    
    # 3. Train Model
    # In live trading, we ideally retrain on ALL past data to get best prediction for tomorrow
    print("[2/5] Training Model on Full History...")
    model_trainer = ModelTrainer()
    
    # Fit model (Note: ModelTrainer uses 'train' segment. 
    # We might want to override train segment to be 'all history' or 'rolling window'.
    # By default it follows dataset config. Ideally we slide the train window to END.
    # But Qlib Dataset spltting is fixed by config. 
    # For now, we assume the model trains on the training set defined in data.py (2015-2023)
    # BUT for Live, we should really train on 2015-Now.
    # Let's see if we can perform a dynamic "Fit" on the whole dataframe?
    # Qlib Model `fit` takes a Dataset.
    # Let's rely on the pre-trained model for now (fast), OR define a new rolling dataset?
    # Simple approach: Train on the standard training set. 
    # Better approach: To be accurate, we should Retrain.
    # Let's just run standard training for now to keep it consistent with backtest.
    # (Improvement: Update train_period in data.py dynamically, but that edits code).
    model_trainer.train(dataset)
    
    # 4. Predict (Inference)
    print("[3/5] Generating Predictions...")
    pred = model_trainer.predict(dataset)
    
    # 5. Signal Processing (EWMA)
    print("[4/5] Applying 10-day EWMA Smoothing (Apex Config)...")
    if pred.index.names[1] == 'instrument':
        level_name = 'instrument'
    else:
        level_name = pred.index.names[1]
        
    pred = pred.groupby(level=level_name).apply(
        lambda x: x.ewm(halflife=10, min_periods=1).mean()
    )
    
    # Clean index (same fix as run_etf_analysis.py)
    if pred.index.nlevels > 2:
        pred = pred.droplevel(0)
    if pred.index.names[0] != 'datetime' and 'datetime' in pred.index.names:
         pred = pred.swaplevel()
    pred = pred.dropna().sort_index()
    
    # 6. Get Latest Date Signals
    latest_date = pred.index.get_level_values('datetime').max()
    print(f"\n[5/5] Latest Signal Date: {latest_date.strftime('%Y-%m-%d')}")
    
    latest_pred = pred.loc[latest_date]
    latest_pred = latest_pred.sort_values(ascending=False)
    
    # --- Market Regime Check (Live) ---
    print("\n[Market Regime Check]")
    from qlib.data import D
    from delorean.config import BENCHMARK
    
    # Fetch last 300 days of benchmark to be safe
    # Qlib 'start_time' needs string or pd.Timestamp
    # Using relative date
    bench_start_check = latest_date - pd.Timedelta(days=400)
    bench_df = D.features([BENCHMARK], ['$close'], start_time=bench_start_check, end_time=latest_date)
    
    if not bench_df.empty:
        bench_close = bench_df.droplevel(0)
        bench_close.columns = ['close']
        bench_close['ma60'] = bench_close['close'].rolling(window=60).mean()
        
        # Latest regime
        is_bull = True # Default
        last_close = 0.0
        last_ma60 = 0.0
        
        try:
            last_close = float(bench_close.loc[latest_date, 'close'])
            last_ma60 = float(bench_close.loc[latest_date, 'ma60'])
            is_bull = last_close > last_ma60
            
            print(f"Benchmark ({BENCHMARK}) Close: {last_close:.2f}")
            print(f"Benchmark MA60: {last_ma60:.2f}")
            
            if not is_bull:
                print("\n" + "!"*40)
                print("WARNING: BEAR MARKET DETECTED (Price < MA60)")
                print("STRATEGY RECOMMENDATION: LIQUIDATE ALL (CASH)")
                print("!"*40)
            else:
                print("Status: BULL Market (Price > MA60). Trading Active.")
        except KeyError:
             print("Warning: Could not calculate MA60 for latest date (Data missing?). Assuming Bull.")
    else:
        print("Warning: Benchmark data not found.")
        is_bull = True
        last_close = 0.0
        last_ma60 = 0.0

    
    # [NEW] Save Daily Recommendations Artifact for Frontend
    rec_artifact = {
        "date": latest_date.strftime('%Y-%m-%d'),
        "market_status": "Bear" if (bench_df.empty or not is_bull) else "Bull",
        "market_data": {
            "benchmark_close": last_close,
            "benchmark_ma60": last_ma60
        },
        "top_recommendations": [],
        "full_rankings": []
    }
    
    # Top K
    for symbol, score in latest_pred.head(topk).items():
        rec_artifact["top_recommendations"].append({
            "symbol": symbol,
            "score": float(score)
        })
        
    # Full list (limit to top 50 to avoid huge json if needed, or all)
    for symbol, score in latest_pred.items():
         rec_artifact["full_rankings"].append({
            "symbol": symbol,
            "score": float(score)
        })
        
    import json
    with open("daily_recommendations.json", "w") as f:
        json.dump(rec_artifact, f, indent=2)
    print(f"\n[Artifact] Saved recommendations to 'daily_recommendations.json'")

    print("\n" + "-"*30)
    print(f"  Top {topk} Recommendations")
    print("-" * 30)
    for i, (symbol, score) in enumerate(latest_pred.head(topk).items(), 1):
        # Determine actionable name (optional mapping)
        print(f"  #{i}  {symbol:<10} (Score: {score:.4f})")
    print("-" * 30)
    
    print("\nFull Rankings (for manual turnover check):")
    print(latest_pred)
    
    print("\n[Strategy Note]")
    print(f"- Target Hold: Top {topk}")
    print(f"- Turnover Control: Only swap if you hold a low-ranked ETF.")
    print("- Aggressive Filter: 'n_drop=1'.")
    print("- Regime Filter: If Bear Market Warning above, prefer CASH.")

if __name__ == "__main__":
    get_trading_signal()
