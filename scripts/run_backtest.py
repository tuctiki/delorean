import sys
import os
import pandas as pd
import datetime
import qlib
from qlib.workflow import R
from qlib.contrib.evaluate import risk_analysis
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from delorean.config import QLIB_PROVIDER_URI, QLIB_REGION, ETF_NAME_MAP
from delorean.data import ETFDataLoader
from delorean.model import ModelTrainer
from delorean.backtest import BacktestEngine

def run_backtest():
    # 1. Initialize Qlib
    qlib.init(provider_uri=QLIB_PROVIDER_URI, region=QLIB_REGION)
    
    print("="*60)
    print("  Delorean: Long-Term Backtest (2015-01-01 to Present)")
    print("="*60)

    # 2. Define Period
    # "Past 3 Years" -> 2023, 2024, 2025 (and Jan 2026)
    train_end = "2022-12-31"
    test_start = "2023-01-01" 
    
    print(f"Training Period: Start -> {train_end}")
    print(f"Testing  Period: {test_start} -> Present")
    print("Note: Early period (2015-2022) is In-Sample (IS). Late period (2023+) is Out-of-Sample (OOS).")
    
    # 3. Load Data
    print("\n[1/4] Loading Data...")
    data_loader = ETFDataLoader(label_horizon=1)
    # We load standard train set and FULL test set
    dataset = data_loader.load_data(train_end=train_end, test_start=test_start)
    
    # 4. Train Model
    print("\n[2/4] Training Model...")
    trainer = ModelTrainer()
    trainer.train(dataset)
    
    # 5. Predict
    print("\n[3/4] Generating Signals (Predictions)...")
    pred = trainer.predict(dataset)
    
    # 6. Run Evaluation
    print("\n[4/4] Running Backtest Engine...")
    
    # Experiment Context for Logging
    import mlflow
    if mlflow.active_run():
        mlflow.end_run()

    with R.start(experiment_name="3year_backtest_v2"):
        R.log_params(
            train_end=train_end,
            test_start=test_start,
            strategy="SimpleTopk",
            topk=5,
            note="3 Year Backtest (2023-Present)"
        )
        
        # Engine execution
        engine = BacktestEngine(pred)
        report, positions = engine.run(topk=5)
        
        # Risk Analysis
        analysis = risk_analysis(report["return"], freq="day")
        
        # Print Summary
        print("\n" + "-"*40)
        print("  Backtest Results (2015 - Present)")
        print("-" * 40)
        
        annual_return = analysis.loc['annualized_return', 'risk']
        max_drawdown = analysis.loc['max_drawdown', 'risk']
        sharpe = analysis.loc['information_ratio', 'risk']
        
        print(f"Annualized Return: {annual_return*100:.2f}%")
        print(f"Max Drawdown:      {max_drawdown*100:.2f}%")
        print(f"Sharpe Ratio:      {sharpe:.2f}")
        print("-" * 40)
        
        # Log Metrics
        R.log_metrics(
            annual_return=annual_return,
            max_drawdown=max_drawdown,
            sharpe=sharpe
        )
        
        # Save Report
        report_path = "artifacts/backtest_report_2015.csv"
        report.to_csv(report_path)
        print(f"\nDetailed report saved to: {report_path}")

        # [NEW] Generate and Log Plot
        print("Generating Cumulative Return Plot...")
        # report['return'] is daily return. 
        # Cumulative = (1 + r).cumprod()
        cum_ret = (1 + report["return"]).cumprod()
        # Benchmark?
        # We can fetch benchmark data to compare
        try:
             from qlib.data import D
             bench_df = D.features([QLIB_PROVIDER_URI] if "510300" in QLIB_PROVIDER_URI else ["510300.SH"], ["$close/$open-1"], start_time=test_start, end_time=pd.Timestamp.now().strftime("%Y-%m-%d"))
             if not bench_df.empty:
                 bench_ret = bench_df.loc(axis=0)[:, "$close/$open-1"].reset_index(0, drop=True)
                 # Align
                 bench_ret = bench_ret.reindex(cum_ret.index).fillna(0)
                 bench_cum = (1 + bench_ret).cumprod()
        except:
             bench_cum = None

        plt.figure(figsize=(12, 6))
        plt.plot(cum_ret.index, cum_ret.values, label="Strategy", color="#58a6ff", linewidth=2)
        if bench_cum is not None:
             plt.plot(cum_ret.index, bench_cum.values, label="Benchmark (HS300)", color="#8b949e", linestyle="--", alpha=0.7)
        
        plt.title(f"Cumulative Return ({test_start} - Present)")
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.legend()
        plt.xlabel("Date")
        plt.ylabel("Normalized Equity")
        
        # Format Date Axis
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.YearLocator())

        plot_path = "artifacts/cumulative_return.png"
        plt.savefig(plot_path)
        plt.close()
        
        # Log to MLflow
        R.log_artifact(plot_path)
        print(f"Plot saved and logged to: {plot_path}")

        # [NEW] Generate Excess Return Plot
        if bench_cum is not None:
             # Ensure alignment
             common_idx = cum_ret.index.intersection(bench_cum.index)
             excess = cum_ret.loc[common_idx] - bench_cum.loc[common_idx]
             
             plt.figure(figsize=(12, 6))
             plt.plot(excess.index, excess.values, label="Excess Return", color="#2ecc71", linewidth=2)
             plt.fill_between(excess.index, excess.values, 0, alpha=0.1, color="#2ecc71")
             plt.title(f"Excess Return vs Benchmark ({test_start} - Present)")
             plt.grid(True, linestyle="--", alpha=0.3)
             plt.legend()
             plt.xlabel("Date")
             plt.ylabel("Excess Equity")
             
             plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
             plt.gca().xaxis.set_major_locator(mdates.YearLocator())
             
             excess_path = "artifacts/excess_return.png"
             plt.savefig(excess_path)
             plt.close()
             
             R.log_artifact(excess_path)
             print(f"Excess Plot saved and logged to: {excess_path}")

if __name__ == "__main__":
    run_backtest()
