#!/usr/bin/env python
"""
Multi-Seed Validation Script

Tests model stability by running backtests with different random seeds.
This helps assess whether performance is robust or just lucky with one seed.

Usage:
    conda run -n quant python scripts/validate_multi_seed.py --seeds 5
    conda run -n quant python scripts/validate_multi_seed.py --seeds 10
"""

import argparse
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import qlib
from delorean.config import (
    QLIB_PROVIDER_URI, QLIB_REGION, BENCHMARK,
    START_TIME, END_TIME, TRAIN_END_TIME, TEST_START_TIME
)
from delorean.data import ETFDataLoader
from delorean.model import ModelTrainer
from delorean.backtest import BacktestEngine
from delorean.signals import smooth_predictions
from delorean.utils import fix_seed, calculate_rank_ic
from qlib.contrib.evaluate import risk_analysis


def run_single_seed(
    seed: int,
    topk: int = 4,
    label_horizon: int = 1,
    smooth_window: int = 10,
    train_end: str = TRAIN_END_TIME,
    test_start: str = TEST_START_TIME,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Run a single backtest with a specific seed.
    
    Returns:
        Dict with sharpe, rank_ic, return, max_dd, and other metrics.
    """
    fix_seed(seed)
    
    # Data Loading
    data_loader = ETFDataLoader(
        label_horizon=label_horizon,
        start_time=START_TIME,
        end_time=END_TIME
    )
    
    dataset = data_loader.load_data(
        train_start=START_TIME,
        train_end=train_end,
        test_start=test_start,
        test_end=END_TIME
    )
    
    # Model Training
    model_trainer = ModelTrainer(seed=seed)
    model_trainer.train(dataset)
    pred = model_trainer.predict(dataset)
    
    # Signal Smoothing
    pred = smooth_predictions(pred, halflife=smooth_window)
    
    # Slice predictions to test period
    ts_start = pd.Timestamp(test_start)
    max_data_date = pred.index.get_level_values('datetime').max()
    pred = pred.loc[ts_start:max_data_date]
    
    # Backtest
    backtest_engine = BacktestEngine(pred)
    report, positions = backtest_engine.run(
        topk=topk,
        drop_rate=0.96,
        n_drop=1,
        buffer=2,
        start_time=test_start,
        end_time=None
    )
    
    # Calculate Metrics
    risk_df = risk_analysis(report['return'])
    
    # Qlib uses 'information_ratio' for Sharpe-like metric
    sharpe = risk_df.loc['information_ratio', 'risk'] if 'information_ratio' in risk_df.index else None
    
    # Annual return and max drawdown
    ann_return = risk_df.loc['annualized_return', 'risk'] if 'annualized_return' in risk_df.index else None
    max_dd = risk_df.loc['max_drawdown', 'risk'] if 'max_drawdown' in risk_df.index else None
    
    # Rank IC
    labels = dataset.prepare("test", col_set="label", data_key="infer")
    rank_ic = calculate_rank_ic(pred, labels)
    
    return {
        'seed': seed,
        'sharpe': sharpe,
        'rank_ic': rank_ic,
        'annual_return': ann_return,
        'max_drawdown': max_dd,
    }


def run_multi_seed_validation(
    n_seeds: int = 5,
    seed_start: int = 1,
    **kwargs
) -> pd.DataFrame:
    """
    Run validation across multiple seeds and return summary statistics.
    """
    results = []
    seeds = list(range(seed_start, seed_start + n_seeds))
    
    print(f"\n{'='*60}")
    print(f"Multi-Seed Validation: Testing {n_seeds} seeds")
    print(f"{'='*60}\n")
    
    for i, seed in enumerate(seeds):
        print(f"[{i+1}/{n_seeds}] Running seed={seed}...", end=" ")
        try:
            result = run_single_seed(seed=seed, **kwargs)
            results.append(result)
            print(f"Sharpe={result['sharpe']:.3f}, IC={result['rank_ic']:.4f}")
        except Exception as e:
            print(f"FAILED: {e}")
            results.append({
                'seed': seed,
                'sharpe': None,
                'rank_ic': None,
                'annual_return': None,
                'max_drawdown': None,
            })
    
    df = pd.DataFrame(results)
    return df


def print_summary(df: pd.DataFrame) -> None:
    """Print summary statistics from multi-seed results."""
    print(f"\n{'='*60}")
    print("SUMMARY STATISTICS")
    print(f"{'='*60}")
    
    valid = df.dropna(subset=['sharpe'])
    
    if len(valid) == 0:
        print("No valid results!")
        return
    
    print(f"\nSeeds tested: {len(df)}")
    print(f"Valid runs: {len(valid)}")
    
    print(f"\n{'Metric':<20} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print("-" * 60)
    
    for col in ['sharpe', 'rank_ic', 'annual_return', 'max_drawdown']:
        if col in valid.columns:
            mean = valid[col].mean()
            std = valid[col].std()
            min_val = valid[col].min()
            max_val = valid[col].max()
            print(f"{col:<20} {mean:>10.4f} {std:>10.4f} {min_val:>10.4f} {max_val:>10.4f}")
    
    # Stability assessment
    sharpe_cv = valid['sharpe'].std() / valid['sharpe'].mean() if valid['sharpe'].mean() != 0 else float('inf')
    print(f"\nSharpe Coefficient of Variation: {sharpe_cv:.2%}")
    
    if sharpe_cv < 0.1:
        stability = "EXCELLENT (CV < 10%)"
    elif sharpe_cv < 0.2:
        stability = "GOOD (CV < 20%)"
    elif sharpe_cv < 0.3:
        stability = "MODERATE (CV < 30%)"
    else:
        stability = "POOR (CV >= 30%)"
    
    print(f"Stability Assessment: {stability}")
    
    # Individual results
    print(f"\n{'='*60}")
    print("INDIVIDUAL RESULTS")
    print(f"{'='*60}")
    print(valid.to_string(index=False))


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-Seed Validation")
    parser.add_argument("--seeds", type=int, default=5, help="Number of seeds to test")
    parser.add_argument("--seed_start", type=int, default=1, help="Starting seed value")
    parser.add_argument("--topk", type=int, default=4, help="TopK holdings")
    parser.add_argument("--label_horizon", type=int, default=1, help="Label horizon days")
    parser.add_argument("--smooth_window", type=int, default=10, help="Signal smoothing window")
    parser.add_argument("--output", type=str, default=None, help="Output CSV path")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Initialize Qlib
    qlib.init(provider_uri=QLIB_PROVIDER_URI, region=QLIB_REGION, kernels=1)
    
    # Suppress verbose output during multi-seed runs
    import logging
    logging.getLogger('qlib').setLevel(logging.WARNING)
    
    # Run validation
    df = run_multi_seed_validation(
        n_seeds=args.seeds,
        seed_start=args.seed_start,
        topk=args.topk,
        label_horizon=args.label_horizon,
        smooth_window=args.smooth_window,
    )
    
    # Print summary
    print_summary(df)
    
    # Save results
    output_path = args.output or f"artifacts/multi_seed_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
