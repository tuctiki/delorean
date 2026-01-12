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

    today = datetime.datetime.now()

    # =========================================================================
    # PHASE 1: Out-of-Sample Validation (The Honest Check)
    # =========================================================================
    print("\n[Phase 1/2] Out-of-Sample Validation (Recent 60 Days)...")
    
    # Validation Split: Train on History[:-60], Test on History[-60:]
    val_days = 60
    val_split_date = today - datetime.timedelta(days=val_days)
    val_train_end = (val_split_date - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
    val_test_start = val_split_date.strftime("%Y-%m-%d")
    
    print(f"  > Training Data: ... to {val_train_end}")
    print(f"  > Test (Validation) Data: {val_test_start} to Present")
    
    # Load Validation Data
    data_loader_val = ETFDataLoader(label_horizon=5)
    dataset_val = data_loader_val.load_data(train_end=val_train_end, test_start=val_test_start)
    
    # Train Validation Model
    print("  > Training Validation Model...")
    model_val = ModelTrainer()
    model_val.train(dataset_val)
    
    # Predict on Validation Set
    print("  > Predicting on Validation Set...")
    pred_val_scores = model_val.predict(dataset_val)
    
    # --- Calculate Validation Metrics (IC & Sharpe) ---
    print("  > Calculating Validation Metrics...")
    
    # Prepare labels for the test segment
    df_val_all = dataset_val.prepare("test", col_set=["label"])
    y_val = df_val_all["label"]
    
    # Align Prediction & Label
    # pred_val_scores index: (datetime, instrument)
    # y_val index: (datetime, instrument)
    val_res = pd.DataFrame({"score": pred_val_scores, "label": y_val.iloc[:, 0]}).dropna()
    
    validation_metrics = {
        "rank_ic": 0.0,
        "sharpe": 0.0,
        "ic_status": "Unknown",
        "sharpe_status": "Unknown"
    }
    
    if not val_res.empty:
        # 1. Rank IC
        daily_ic = val_res.groupby(level='datetime').apply(lambda x: x["score"].corr(x["label"], method="spearman"))
        mean_ic = daily_ic.mean()
        validation_metrics["rank_ic"] = float(mean_ic)
        print(f"    Validation Rank IC: {mean_ic:.4f}") 
        
        if mean_ic < 0.02: # Threshold
            print("!"*40)
            print(f"    WARNING: Low OOS Rank IC ({mean_ic:.4f}). Model predictive power is weak.")
            print("!"*40)
            validation_metrics["ic_status"] = "Warning"
        else:
             print("    [PASS] Rank IC acceptable.")
             validation_metrics["ic_status"] = "Pass"

        # 2. Sharpe (Simulated Backtest)
        print("    Simulating Strategy on Validation period...")
        try:
             engine = BacktestEngine(pred_val_scores)
             report_val, _ = engine.run(topk=topk)
             risks = risk_analysis(report_val["return"], freq="day")
             sharpe = risks.loc["information_ratio", "risk"]
             validation_metrics["sharpe"] = float(sharpe)
             print(f"    Validation Sharpe: {sharpe:.4f}")
             
             if sharpe < 0.0:
                 validation_metrics["sharpe_status"] = "Critical"
                 print("    CRITICAL: Negative Sharpe in recent period.")
             elif sharpe < 0.5:
                 validation_metrics["sharpe_status"] = "Warning"
             else:
                 validation_metrics["sharpe_status"] = "Pass"
        except Exception as e:
            print(f"    Warning: Sharpe calc failed: {e}")
            validation_metrics["sharpe_status"] = "Error"
    else:
        print("    Warning: No validation data available.")

    # =========================================================================
    # PHASE 2: Production Signal Generation (Full History)
    # =========================================================================
    print("\n[Phase 2/2] Production Signal Generation (Full History)...")

    # Production Split: Train on ALL available history to predict Tomorrow.
    # We stick to the robust 14-day split to ensure Qlib has a non-empty 'test' segment 
    # for the most recent days, avoiding any empty-dataframe errors.
    prod_split_date = today - datetime.timedelta(days=14)
    prod_train_end = (prod_split_date - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
    prod_test_start = prod_split_date.strftime("%Y-%m-%d")

    print(f"  > Training Data: ... to {prod_train_end}")
    print(f"  > Prediction Target: {prod_test_start} to Present")
    
    # Load Production Data
    data_loader_prod = ETFDataLoader(label_horizon=5)
    dataset_prod = data_loader_prod.load_data(train_end=prod_train_end, test_start=prod_test_start)
    
    # Train Production Model
    print("  > Training Production Model...")
    model_prod = ModelTrainer()
    model_prod.train(dataset_prod)
    
    # Generate Signals
    print("  > Generating Signals...")
    pred_prod = model_prod.predict(dataset_prod)
    
    # Signal Smoothing (EWMA)
    print("  > Applying 20-day EWMA Smoothing...")
    if pred_prod.index.names[1] == 'instrument':
        level_name = 'instrument'
    else:
        level_name = pred_prod.index.names[1]
        
    pred_smooth = pred_prod.groupby(level=level_name).apply(
        lambda x: x.ewm(halflife=20, min_periods=1).mean()
    )
    
    # Clean Index
    if pred_smooth.index.nlevels > 2:
        pred_smooth = pred_smooth.droplevel(0)
    if pred_smooth.index.names[0] != 'datetime' and 'datetime' in pred_smooth.index.names:
         pred_smooth = pred_smooth.swaplevel()
    pred_smooth = pred_smooth.dropna().sort_index()
    
    # Extract Latest Signal
    latest_date = pred_smooth.index.get_level_values('datetime').max()
    print(f"\n[Result] Latest Signal Date: {latest_date.strftime('%Y-%m-%d')}")
    
    latest_pred = pred_smooth.loc[latest_date]
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
