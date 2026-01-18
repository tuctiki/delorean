#!/usr/bin/env python3
"""
Compare backtest results across different signal_halflife configurations.
"""
import glob
import os

def extract_metrics(exp_dir):
    """Extract metrics from an MLflow run."""
    metrics = {}
    for metric_name in ['annualized_return', 'sharpe', 'max_drawdown', 'ann_turnover', 'rank_ic']:
        path = os.path.join(exp_dir, 'metrics', metric_name)
        if os.path.exists(path):
            with open(path) as f:
                lines = f.readlines()
                if lines:
                    metrics[metric_name] = float(lines[-1].split()[1])
    return metrics

def get_experiment_name(exp_dir):
    """Get experiment name from params."""
    param_path =  os.path.join(os.path.dirname(exp_dir), 'params', 'experiment_name')
    if os.path.exists(param_path):
        with open(param_path) as f:
            return f.read().strip()
    return "Unknown"

# Find recent runs with ann_turnover (our 3 test runs)
runs = sorted(glob.glob('mlruns/*/*/metrics/ann_turnover'), key=os.path.getmtime, reverse=True)[:3]

print("=" * 100)
print("SIGNAL SMOOTHING COMPARISON - Backtest Results (2022-2025)")
print("=" * 100)
print()

results = []
for i, run_path in enumerate(runs):
    exp_dir = os.path.dirname(os.path.dirname(run_path))
    exp_name = get_experiment_name(exp_dir)
    metrics = extract_metrics(os.path.dirname(run_path))
    
    if metrics:
        results.append((exp_name, metrics))

# Sort by experiment name to get consistent ordering
results.sort(key=lambda x: x[0])

print(f"{'Config':<20} {'Ann Return':<12} {'Sharpe':<10} {'Max DD':<12} {'Ann Turnover':<15} {'Rank IC':<10}")
print("-" * 100)

for exp_name, metrics in results:
    print(f"{exp_name:<20} "
          f"{metrics.get('annualized_return', 0)*100:>10.2f}%  "
          f"{metrics.get('sharpe', 0):>9.2f}  "
          f"{metrics.get('max_drawdown', 0)*100:>10.2f}%  "
          f"{metrics.get('ann_turnover', 0)*100:>13.1f}%  "
          f"{metrics.get('rank_ic', 0):>9.3f}")

print()
print("=" * 100)
print("ANALYSIS")
print("=" * 100)

# Find best by different criteria
if results:
    best_sharpe = max(results, key=lambda x: x[1].get('sharpe', 0))
    best_return = max(results, key=lambda x: x[1].get('annualized_return', 0))
    lowest_turnover = min(results, key=lambda x: x[1].get('ann_turnover', 999))
    lowest_dd = max(results, key=lambda x: x[1].get('max_drawdown', -999))  # max because DD is negative
    
    print(f"\n✅ Best Sharpe Ratio: {best_sharpe[0]} ({best_sharpe[1]['sharpe']:.2f})")
    print(f"✅ Best Return: {best_return[0]} ({best_return[1]['annualized_return']*100:.2f}%)")
    print(f"✅ Lowest Turnover: {lowest_turnover[0]} ({lowest_turnover[1]['ann_turnover']*100:.1f}%)")
    print(f"✅ Lowest Drawdown: {lowest_dd[0]} ({lowest_dd[1]['max_drawdown']*100:.2f}%)")
    
    print("\n" + "=" * 100)
    print("RECOMMENDATION")
    print("=" * 100)
    
    # Simple scoring: balance Sharpe, turnover, and drawdown
    for exp_name, metrics in results:
        sharpe = metrics.get('sharpe', 0)
        turnover = metrics.get('ann_turnover', 10)  # 1000% default if missing
        dd = abs(metrics.get('max_drawdown', -0.5))
        
        # Higher score is better
        # Sharpe weight = 2, turnover penalty, drawdown penalty
        score = sharpe * 2 - (turnover / 100) * 0.5 - dd * 5
        print(f"{exp_name}: Score = {score:.2f}")
