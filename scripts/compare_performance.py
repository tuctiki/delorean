#!/usr/bin/env python3
"""
Compare performance before and after adding new factors.
"""
import mlflow
import pandas as pd
from datetime import datetime

mlflow.set_tracking_uri('mlruns/')
client = mlflow.tracking.MlflowClient()

# Get the default experiment
experiment = client.get_experiment_by_name('ETF_Strategy')

if experiment:
    # Get recent runs
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=['start_time DESC'],
        max_results=10
    )
    
    print("=" * 80)
    print("RECENT BACKTEST RUNS")
    print("=" * 80)
    
    results = []
    for run in runs:
        start_time = pd.to_datetime(run.info.start_time, unit='ms')
        metrics = run.data.metrics
        params = run.data.params
        
        results.append({
            'Run ID': run.info.run_id[:8],
            'Start Time': start_time.strftime('%Y-%m-%d %H:%M'),
            'Sharpe': metrics.get('sharpe', 0),
            'Rank IC': metrics.get('rank_ic', 0),
            'L2 Train': metrics.get('l2.train', 0),
            'Label Horizon': params.get('label_horizon', 'N/A'),
            'Smooth Window': params.get('smooth_window', 'N/A'),
            'TopK': params.get('topk', 'N/A'),
        })
    
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    
    # Highlight the latest run
    if len(results) > 0:
        latest = results[0]
        print(f"\n{'=' * 80}")
        print(f"LATEST RUN (With New Factors)")
        print(f"{'=' * 80}")
        print(f"Run ID: {latest['Run ID']}")
        print(f"Time: {latest['Start Time']}")
        print(f"Sharpe Ratio: {latest['Sharpe']:.4f}")
        print(f"Rank IC: {latest['Rank IC']:.4f}")
        print(f"L2 Train: {latest['L2 Train']:.6f}")
        
        # Compare with previous runs
        if len(results) > 1:
            print(f"\n{'=' * 80}")
            print("COMPARISON WITH PREVIOUS RUNS")
            print(f"{'=' * 80}")
            
            prev_sharpes = [r['Sharpe'] for r in results[1:] if r['Sharpe'] > 0]
            if prev_sharpes:
                avg_prev_sharpe = sum(prev_sharpes) / len(prev_sharpes)
                sharpe_change = latest['Sharpe'] - avg_prev_sharpe
                print(f"Average Previous Sharpe: {avg_prev_sharpe:.4f}")
                print(f"Current Sharpe: {latest['Sharpe']:.4f}")
                print(f"Change: {sharpe_change:+.4f} ({sharpe_change/avg_prev_sharpe*100:+.2f}%)")
            
            prev_ics = [r['Rank IC'] for r in results[1:] if r['Rank IC'] != 0]
            if prev_ics:
                avg_prev_ic = sum(prev_ics) / len(prev_ics)
                ic_change = latest['Rank IC'] - avg_prev_ic
                print(f"\nAverage Previous Rank IC: {avg_prev_ic:.4f}")
                print(f"Current Rank IC: {latest['Rank IC']:.4f}")
                print(f"Change: {ic_change:+.4f} ({ic_change/avg_prev_ic*100:+.2f}%)")

print(f"\n{'=' * 80}")
print("Analysis complete!")
print(f"{'=' * 80}")
