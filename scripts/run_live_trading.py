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
from qlib.data import D

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
    print("  > Applying 15-day EWMA Smoothing...")
    if pred_prod.index.names[1] == 'instrument':
        level_name = 'instrument'
    else:
        level_name = pred_prod.index.names[1]
        
    pred_smooth = pred_prod.groupby(level=level_name).apply(
        lambda x: x.ewm(halflife=15, min_periods=1).mean()
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
    
    # --- Market Regime Check (Global HS300) ---
    print("\n[Market Regime Check] Global HS300 Filter (Price > MA60)...")
    from delorean.config import BENCHMARK, ETF_LIST, ETF_NAME_MAP
    
    # 1. Fetch Benchmark Data
    bench_df = D.features([BENCHMARK], ['$close'], start_time=latest_date, end_time=latest_date)
    bench_close = 0.0
    bench_ma60 = 0.0
    is_bull = True
    
    if not bench_df.empty:
        # Fetch Close and MA60 for Benchmark
        bench_feat = D.features([BENCHMARK], ['$close', 'Mean($close, 60)'], start_time=latest_date, end_time=latest_date)
        if not bench_feat.empty:
             bench_close = bench_feat.iloc[0, 0]
             bench_ma60 = bench_feat.iloc[0, 1]
             
             if bench_close < bench_ma60:
                 is_bull = False
                 print(f"  [BEAR DETECTED] HS300 Close ({bench_close:.2f}) < MA60 ({bench_ma60:.2f}). Liquidate All.")
             else:
                 print(f"  [BULL CONFIRMED] HS300 Close ({bench_close:.2f}) > MA60 ({bench_ma60:.2f}). Trade On.")
    
    # [NEW] Configuration Artifact
    strategy_config = {
        "topk": topk,
        "smooth_window": 15,
        "buffer": 2,
        "label_horizon": 5,
        "mode": "Equal Weight + Global HS300 Filter"
    }

    rec_artifact = {
        "date": latest_date.strftime('%Y-%m-%d'),
        "generation_time": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "market_status": "Bull" if is_bull else "Bear",
        "validation": validation_metrics,
        "market_data": {
            "benchmark_close": float(bench_close),
            "benchmark_ma60": float(bench_ma60)
        },
        "strategy_config": strategy_config,
        "top_recommendations": [],
        "buffer_holdings": [], 
        "full_rankings": []
    }

    # 2. Populate Recommendations
    # Fetch Volatility for display
    fields = [
        '$close', 
        'Std($close/Ref($close,1)-1, 20)'
    ]
    names = ['close', 'vol20']
    
    feat_df = D.features(ETF_LIST, fields, start_time=latest_date, end_time=latest_date)
    feat_df.columns = names
    
    vol_map = {}
    close_map = {}
    
    if not feat_df.empty:
        # Reset index
         if 'datetime' in feat_df.index.names:
             feat_reset = feat_df.droplevel('datetime')
         else:
             feat_reset = feat_df
         vol_map = feat_reset['vol20'].to_dict()
         close_map = feat_reset['close'].to_dict()

    if is_bull:
        # Bull Market: Top K + Buffer
        # We include topk + 2 items. The extra 2 are "buffer" items.
        buffer_size = 2
        for i, (symbol, score) in enumerate(latest_pred.head(topk + buffer_size).items(), 1):
             vol_raw = vol_map.get(symbol, 0.0)
             close_price = close_map.get(symbol, 0.0)
             
             is_buffer = i > topk
             
             # Weight Calculation
             # Buffer items get 0 weight initially (frontend can display them as holding candidates)
             # Top K items get 1/K weight
             weight = 1.0 / topk if not is_buffer else 0.0
             
             item = {
                "rank": i,
                "symbol": symbol,
                "name": ETF_NAME_MAP.get(symbol, symbol),
                "score": float(score),
                "volatility": float(vol_raw),
                "current_price": float(close_price),
                "target_weight": float(weight),
                "is_buffer": is_buffer
            }
             rec_artifact["top_recommendations"].append(item)
    else:
        # Bear Market: Empty Recommendations (Hold Cash)
        pass 
    
    # Full Rankings always populated for reference
    for symbol, score in latest_pred.items():
         vol_raw = vol_map.get(symbol, 0.0)
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
