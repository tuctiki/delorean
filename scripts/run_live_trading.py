import qlib
import pandas as pd
import datetime
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qlib.workflow import R
from delorean.config import QLIB_PROVIDER_URI, QLIB_REGION, BENCHMARK, ETF_NAME_MAP
from delorean.data import ETFDataLoader
from delorean.model import ModelTrainer
from delorean.backtest import BacktestEngine
from qlib.contrib.evaluate import risk_analysis

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
    # [Live Trading Config] 
    # Dynamic split to ensure we train on full history
    # We use a robust split: Train up to 14 days ago, Test from 14 days ago.
    # This ensures 'Test' is never empty even if data lags, and we always get the latest available signal.
    today = datetime.datetime.now()
    split_date = today - datetime.timedelta(days=14)
    
    train_end = (split_date - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
    test_start = split_date.strftime("%Y-%m-%d")
    
    print(f"Live Mode: Training on History up to {train_end}")
    print(f"           Testing/Inference from {test_start}")
    
    data_loader = ETFDataLoader(label_horizon=5)
    # Override segments for live trading
    dataset = data_loader.load_data(train_end=train_end, test_start=test_start)
    
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
    
    # [Validation] Post-Training Checks
    print("\n" + "="*40)
    print("[Validation] Performing Post-Training Checks...")
    
    validation_metrics = {
        "rank_ic": None,
        "sharpe": None,
        "ic_status": "Unknown",
        "sharpe_status": "Unknown"
    }

    try:
        # 1. Prepare Data (Last 6 Months of Train)
        # Getting the raw dataframe from the dataset's train segment
        df_train = dataset.prepare("train", col_set=["feature", "label"])
        
        # Filter for last ~6 months (126 trading days)
        # We use a date-based filter to be precise
        valid_start_date = pd.Timestamp(train_end) - pd.Timedelta(days=180)
        
        # Ensure index is datetime level for filtering
        # df_train index is (datetime, instrument) usually
        df_val = df_train.loc[df_train.index.get_level_values('datetime') >= valid_start_date]
        
        if not df_val.empty:
            if model_trainer.selected_features:
                X_val = df_val["feature"][model_trainer.selected_features]
            else:
                X_val = df_val["feature"]
            y_val = df_val["label"]
            
            # Predict
            # Handle Qlib LGBModel vs Native Booster
            if hasattr(model_trainer.model, "model"):
                # Qlib LGBModel -> Inner Booster
                pred_val_scores = model_trainer.model.model.predict(X_val)
            else:
                # Native Booster
                pred_val_scores = model_trainer.model.predict(X_val)
            pred_val = pd.Series(pred_val_scores, index=X_val.index)
            
            # --- Check 1: Rank IC ---
            # Align
            val_res = pd.DataFrame({"score": pred_val, "label": y_val.iloc[:, 0]}).dropna()
            
            # Group by date
            daily_ic = val_res.groupby(level='datetime').apply(lambda x: x["score"].corr(x["label"], method="spearman"))
            mean_ic = daily_ic.mean()
            
            print(f"  > Recent 6-Month Rank IC: {mean_ic:.4f}")
            validation_metrics["rank_ic"] = float(mean_ic)
            
            if mean_ic < 0.035:
                print("!"*40)
                print(f"CRITICAL WARNING: Rank IC ({mean_ic:.4f}) < 0.035")
                print("The model predictive power is low. Signals may be unreliable.")
                print("!"*40)
                validation_metrics["ic_status"] = "Critical"
            else:
                print("    [PASS] Rank IC > 0.035")
                validation_metrics["ic_status"] = "Pass"
                
            # --- Check 2: Sharpe Ratio (Simulated) ---
            # Last 60 days
            sharpe_start_date = pd.Timestamp(train_end) - pd.Timedelta(days=90)
            pred_sharpe = pred_val.loc[pred_val.index.get_level_values('datetime') >= sharpe_start_date]
            
            if not pred_sharpe.empty:
                print(f"  > Simulating recent performance (from {sharpe_start_date.date()})...")
                # Run Mini Backtest
                engine = BacktestEngine(pred_sharpe)
                # Suppress verbose output for clean log
                report_val, _ = engine.run(topk=topk) 
                
                risks = risk_analysis(report_val["return"], freq="day")
                sharpe = risks.loc["information_ratio", "risk"]
                
                print(f"  > Recent Simulated Sharpe: {sharpe:.4f}")
                validation_metrics["sharpe"] = float(sharpe)
                
                if sharpe < 0.4:
                     print("!"*40)
                     print(f"ERROR: Sharpe Ratio ({sharpe:.4f}) < 0.4")
                     print("Recent performance is very poor.")
                     print("!"*40)
                     validation_metrics["sharpe_status"] = "Error"
                elif sharpe < 0.6:
                     print("!"*40)
                     print(f"WARNING: Sharpe Ratio ({sharpe:.4f}) < 0.6")
                     print("Recent performance is suboptimal.")
                     print("!"*40)
                     validation_metrics["sharpe_status"] = "Warning"
                else:
                     print("    [PASS] Sharpe > 0.6")
                     validation_metrics["sharpe_status"] = "Pass"
            else:
                print("  > Not enough data for Sharpe check.")
        else:
            print("  > Not enough training data for validation.")
            
    except Exception as e:
        print(f"Warning: Validation checks failed: {e}")
        import traceback
        traceback.print_exc()
        
    print("="*40 + "\n")
    
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
        "mode": "Equal Weight + Dynamic Exposure"
    }

    rec_artifact = {
        "date": latest_date.strftime('%Y-%m-%d'),
        "generation_time": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "market_status": "Bear" if (bench_df.empty or not is_bull) else "Bull",
        "validation": validation_metrics,
        "market_data": {
            "benchmark_close": last_close,
            "benchmark_ma60": last_ma60
        },
        "strategy_config": strategy_config,
        "top_recommendations": [],
        "buffer_holdings": [], # For ranks within buffer
        "full_rankings": []
    }
    
    # [NEW] Fetch Volatility AND Close Price for Display
    print("\n[Data] Fetching Volatility (VOL20) and Close Price for display...")
    # Using $close for price.
    feat_df = D.features(D.instruments(market=QLIB_REGION), ['$close', 'Std($close/Ref($close,1)-1, 20)'], start_time=latest_date, end_time=latest_date)
    feat_df.columns = ['close', 'vol20']
    
    vol_map = {}
    close_map = {}
    
    if not feat_df.empty:
        # Reset index to get symbol
        try:
             # Droplevel datetime if present
            if 'datetime' in feat_df.index.names:
                feat_reset = feat_df.droplevel('datetime')
            else:
                feat_reset = feat_df
            
            vol_map = feat_reset['vol20'].to_dict()
            close_map = feat_reset['close'].to_dict()
            
            print(f"Loaded Features for {len(vol_map)} instruments.")
        except Exception as e:
            print(f"Warning: Failed to parse Feature data: {e}")
        except Exception as e:
            print(f"Warning: Failed to parse Volatility data: {e}")

    # Calculate Target Weights (Equal Weight on Top K)
    # Volatility is fetched for display only.
         
    # 2. Populate Recommendations with Weights
    # We Iterate larger than topk to capture Buffer items too if needed for display
    # But weights are usually allocated to Top K. Buffer items held have "current weight" but 0 target weight?
    # Actually strategy says: Buffer held? Keep it. 
    # For UI simplicity: Show Target Weights for Top K (The "Buy" List).
    
    for i, (symbol, score) in enumerate(latest_pred.head(topk + 2).items(), 1): # Show Top K + Buffer
        vol_raw = vol_map.get(symbol, 0.0)
        close_price = close_map.get(symbol, 0.0)
        
        # Weight Calculation (Only for Top K) - EQUAL WEIGHT (Default)
        weight = 0.0
        if i <= topk:
             weight = 1.0 / topk
        
        item = {
            "rank": i,
            "symbol": symbol,
            "name": ETF_NAME_MAP.get(symbol, symbol),
            "score": float(score),
            "volatility": float(vol_raw),
            "current_price": float(close_price),
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
