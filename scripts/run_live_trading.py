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

def get_trading_signal(topk=5):
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
    data_loader = ETFDataLoader(label_horizon=5)
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
    print("[4/5] Applying 20-day EWMA Smoothing (Apex Config)...")
    if pred.index.names[1] == 'instrument':
        level_name = 'instrument'
    else:
        level_name = pred.index.names[1]
        
    pred = pred.groupby(level=level_name).apply(
        lambda x: x.ewm(halflife=20, min_periods=1).mean()
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

    
    # [NEW] Configuration Artifact
    # We should match this with how we ran the script, or just hardcode the "Active" settings
    # Since this script "is" the active strategy, we define what it does.
    strategy_config = {
        "topk": topk,
        "smooth_window": 20,
        "buffer": 2,
        "label_horizon": 5,
        "mode": "Risk Parity + Dynamic Exposure"
    }

    rec_artifact = {
        "date": latest_date.strftime('%Y-%m-%d'),
        "generation_time": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "market_status": "Bear" if (bench_df.empty or not is_bull) else "Bull",
        "market_data": {
            "benchmark_close": last_close,
            "benchmark_ma60": last_ma60
        },
        "strategy_config": strategy_config,
        "top_recommendations": [],
        "buffer_holdings": [], # For ranks within buffer
        "full_rankings": []
    }
    
    # [NEW] Fetch Volatility for Display
    print("\n[Data] Fetching Volatility (VOL20) for display...")
    vol_df = D.features(D.instruments(market=QLIB_REGION), ['Std($close/Ref($close,1)-1, 20)'], start_time=latest_date, end_time=latest_date)
    # vol_df index: (instrument, datetime) since single day? Or just (instrument, datetime).
    # Check shape
    vol_map = {}
    if not vol_df.empty:
        # Reset index to get symbol
        # Typically looks like: 
        #                      Std...
        # instrument datetime        
        # SH510300   2023-01-01  0.012
        
        # We want simple map: symbol -> vol
        try:
             # Droplevel datetime if present
            if 'datetime' in vol_df.index.names:
                vol_reset = vol_df.droplevel('datetime')
            else:
                vol_reset = vol_df
            
            vol_map = vol_reset.iloc[:, 0].to_dict() # symbol -> vol float
        except Exception as e:
            print(f"Warning: Failed to parse Volatility data: {e}")

    # Calculate Target Weights (Risk Parity on Top K)
    # 1. Get Vols for Top K
    topk_candidates = latest_pred.head(topk).index
    inv_vols = {}
    sum_inv_vol = 0.0
    
    for symbol in topk_candidates:
         vol_raw = vol_map.get(symbol, 0.0)
         if pd.isna(vol_raw) or vol_raw <= 0: vol_raw = 0.01 # Avoid div by zero, assume low risk/avg
         inv_vol = 1.0 / vol_raw
         inv_vols[symbol] = inv_vol
         sum_inv_vol += inv_vol
         
    # 2. Populate Recommendations with Weights
    # We Iterate larger than topk to capture Buffer items too if needed for display
    # But weights are usually allocated to Top K. Buffer items held have "current weight" but 0 target weight?
    # Actually strategy says: Buffer held? Keep it. 
    # For UI simplicity: Show Target Weights for Top K (The "Buy" List).
    
    for i, (symbol, score) in enumerate(latest_pred.head(topk + 2).items(), 1): # Show Top K + Buffer
        vol_raw = vol_map.get(symbol, 0.0)
        
        # Weight Calculation (Only for Top K)
        weight = 0.0
        if i <= topk:
            if sum_inv_vol > 0:
                 weight = inv_vols.get(symbol, 0.0) / sum_inv_vol
            else:
                 weight = 1.0 / topk
        
        item = {
            "rank": i,
            "symbol": symbol,
            "score": float(score),
            "volatility": float(vol_raw),
            "target_weight": float(weight),
            "is_buffer": i > topk
        }
        
        rec_artifact["top_recommendations"].append(item)
        
    # Full list (limit to top 50 to avoid huge json if needed, or all)
    for symbol, score in latest_pred.items():
         vol_raw = vol_map.get(symbol, 0.0)
         if pd.isna(vol_raw): vol_raw = 0.0
         rec_artifact["full_rankings"].append({
            "symbol": symbol,
            "score": float(score),
            "volatility": float(vol_raw)
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
    print(f"- Buffer Logic (Hysteresis): Keep existing holdings if Rank <= {topk+2}.")
    print(f"- Turnover Control: Only swap if Rank > {topk+2}.")
    print("- Regime Filter: If Bear Market Warning above, prefer CASH.")

if __name__ == "__main__":
    get_trading_signal()
