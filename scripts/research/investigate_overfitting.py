import qlib
import pandas as pd
import sys
import os
import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from delorean.config import QLIB_PROVIDER_URI, QLIB_REGION, ETF_LIST
from delorean.data import ETFDataLoader
from delorean.model import ModelTrainer
from delorean.backtest import BacktestEngine
from delorean.utils import smooth_predictions, calculate_rank_ic
from qlib.contrib.evaluate import risk_analysis

def run_investigation():
    # 1. Initialize Qlib
    qlib.init(provider_uri=QLIB_PROVIDER_URI, region=QLIB_REGION)
    print("="*60)
    print("  OVERFITTING INVESTIGATION: 2019-2021 OOS CHECK")
    print("="*60)

    # 2. Define Data Periods (Strict Isolation)
    # Train: 2015-01-01 -> 2018-12-31
    # Test (OOS): 2019-01-01 -> 2021-12-31
    TRAIN_START = "2015-01-01"
    TRAIN_END = "2018-12-31" 
    TEST_START = "2019-01-01"
    TEST_END = "2021-12-31"

    print(f"\n[1] Loading Data...")
    print(f"    Train: {TRAIN_START} -> {TRAIN_END}")
    print(f"    Test:  {TEST_START} -> {TEST_END}")

    # Initialize DataLoader with the full range we need
    # We set start_time/end_time for the handler to ensure it covers everything.
    data_loader = ETFDataLoader(start_time=TRAIN_START, end_time=TEST_END, label_horizon=5)
    
    # Load dataset with specific split
    dataset = data_loader.load_data(
        train_start=TRAIN_START, 
        train_end=TRAIN_END, 
        test_start=TEST_START, 
        test_end=TEST_END
    )

    # 3. Train Model
    print(f"\n[2] Training Model on {TRAIN_START} - {TRAIN_END}...")
    trainer = ModelTrainer(seed=42)
    trainer.train(dataset)

    # 4. Predict on Test Set
    print(f"\n[3] Generating Predictions for {TEST_START} - {TEST_END}...")
    pred_test = trainer.predict(dataset)
    
    # 5. Parameter Sweep (Smoothing Windows)
    # Current config uses window=15. We test [5, 10, 15, 20]
    windows = [5, 10, 15, 20]
    results = []

    print(f"\n[4] Running Parameter Sweep (Smoothing Windows: {windows})...")
    
    for window in windows:
        print(f"\n  >>> Testing Smooth Window = {window} ...")
        
        # Apply EWMA Smoothing using centralized utility
        pred_smooth = smooth_predictions(pred_test, halflife=window)

        # Run Backtest
        engine = BacktestEngine(pred_smooth)
        
        try:
            # We pass explicit start/end time to match our OOS period exactly
            report, _ = engine.run(
                topk=3, 
                start_time=TEST_START, 
                end_time=TEST_END, 
                market_regime=None
            )
            
            # Analyze Metrics
            risks = risk_analysis(report["return"], freq="day")
            
            # Key Metrics
            ret_annual = risks.loc["annualized_return", "risk"]
            sharpe = risks.loc["information_ratio", "risk"]
            mdd = risks.loc["max_drawdown", "risk"]
            
            # Rank IC using centralized utility
            df_test_all = dataset.prepare("test", col_set=["label"])
            y_test = df_test_all["label"]
            ic = calculate_rank_ic(pred_smooth, y_test.iloc[:, 0])

            print(f"      [Result] Sharpe: {sharpe:.4f}, Ann. Ret: {ret_annual:.4f}, MDD: {mdd:.4f}, Rank IC: {ic:.4f}")
            
            results.append({
                "window": window,
                "sharpe": sharpe,
                "annualized_return": ret_annual,
                "max_drawdown": mdd,
                "rank_ic": ic
            })

        except Exception as e:
            print(f"      [Error] Backtest failed: {e}")
            import traceback
            traceback.print_exc()

    # 6. Report
    print("\n" + "="*60)
    print("  RESULTS SUMMARY (2019-2021)")
    print("="*60)
    res_df = pd.DataFrame(results)
    if not res_df.empty:
        print(res_df.to_string(index=False))
        
        # Check robustness
        best_sharpe = res_df['sharpe'].max()
        best_window = res_df.loc[res_df['sharpe'].idxmax(), 'window']
        
        print(f"\n  Best Window: {best_window} (Sharpe: {best_sharpe:.4f})")
        
        # Check if 15 (current) is reasonably close to best
        curr_sharpe = res_df.loc[res_df['window'] == 15, 'sharpe'].values[0] if 15 in res_df['window'].values else 0.0
        
        if curr_sharpe > 1.0 and (best_sharpe - curr_sharpe) < 0.5:
             print("  [Conclusion] Window 15 performs reasonably well in OOS. Overfitting unlikely.")
        elif curr_sharpe < 0.5:
             print("  [Conclusion] Window 15 performs POORLY in OOS. Possible Overfitting or Regime Change.")
        else:
             print("  [Conclusion] Results mixed. Check comparative performance.")
            
    else:
        print("  No results generated.")
    print("="*60)

if __name__ == "__main__":
    run_investigation()
