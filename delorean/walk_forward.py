"""
Walk-Forward Validation for ETF Strategy.

Implements rolling train/predict to match live trading behavior
where models are retrained periodically with latest data.
"""

import pandas as pd
import datetime
from typing import Optional, Tuple, List
from dataclasses import dataclass

from .config import ETF_LIST, START_TIME, END_TIME
from .data import ETFDataLoader
from .model import ModelTrainer
from .signals import smooth_predictions


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward validation."""
    train_window_months: int = 24  # Months of training data
    retrain_frequency_months: int = 1  # How often to retrain
    smooth_window: int = 10  # EWMA halflife for smoothing
    label_horizon: int = 1  # Forward return prediction horizon
    seed: int = 42


class WalkForwardValidator:
    """
    Implements walk-forward validation (rolling train/predict).
    
    This matches live trading behavior where models are retrained periodically
    with the latest available data.
    
    Example:
        validator = WalkForwardValidator(config)
        predictions = validator.run(
            test_start='2023-01-01',
            test_end='2026-01-15'
        )
    """
    
    def __init__(self, config: WalkForwardConfig = None):
        """
        Initialize the walk-forward validator.
        
        Args:
            config: Configuration parameters. Uses defaults if not provided.
        """
        self.config = config or WalkForwardConfig()
        self.predictions: List[pd.Series] = []
        self.metrics: List[dict] = []
    
    def _generate_windows(
        self, 
        test_start: str, 
        test_end: str
    ) -> List[Tuple[str, str, str, str]]:
        """
        Generate (train_start, train_end, pred_start, pred_end) tuples.
        
        Args:
            test_start: Start of test period (YYYY-MM-DD)
            test_end: End of test period (YYYY-MM-DD)
            
        Returns:
            List of (train_start, train_end, pred_start, pred_end) tuples
        """
        windows = []
        
        # Parse dates
        start_dt = pd.Timestamp(test_start)
        end_dt = pd.Timestamp(test_end)
        
        current_pred_start = start_dt
        
        while current_pred_start < end_dt:
            # Training window: [current - train_window, current - 1 day]
            train_end = current_pred_start - pd.Timedelta(days=1)
            train_start = train_end - pd.DateOffset(months=self.config.train_window_months)
            
            # Prediction window: [current, current + retrain_frequency]
            pred_end = min(
                current_pred_start + pd.DateOffset(months=self.config.retrain_frequency_months) - pd.Timedelta(days=1),
                end_dt
            )
            
            windows.append((
                train_start.strftime('%Y-%m-%d'),
                train_end.strftime('%Y-%m-%d'),
                current_pred_start.strftime('%Y-%m-%d'),
                pred_end.strftime('%Y-%m-%d')
            ))
            
            # Move to next window
            current_pred_start = pred_end + pd.Timedelta(days=1)
        
        return windows
    
    def run(
        self, 
        test_start: str, 
        test_end: str,
        verbose: bool = True
    ) -> pd.Series:
        """
        Run walk-forward validation.
        
        Args:
            test_start: Start of test period (YYYY-MM-DD)
            test_end: End of test period (YYYY-MM-DD)
            verbose: Whether to print progress
            
        Returns:
            Concatenated predictions for entire test period
        """
        windows = self._generate_windows(test_start, test_end)
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"  Walk-Forward Validation")
            print(f"  Test Period: {test_start} to {test_end}")
            print(f"  Training Window: {self.config.train_window_months} months")
            print(f"  Retrain Frequency: {self.config.retrain_frequency_months} month(s)")
            print(f"  Total Windows: {len(windows)}")
            print(f"{'='*60}\n")
        
        self.predictions = []
        
        for i, (train_start, train_end, pred_start, pred_end) in enumerate(windows, 1):
            if verbose:
                print(f"[Window {i}/{len(windows)}] Train: {train_start} → {train_end}, Predict: {pred_start} → {pred_end}")
            
            # Load data for this window
            data_loader = ETFDataLoader(
                label_horizon=self.config.label_horizon,
                start_time=train_start,
                end_time=pred_end
            )
            
            dataset = data_loader.load_data(
                train_start=train_start,
                train_end=train_end,
                test_start=pred_start,
                test_end=pred_end
            )
            
            # Check if we have data - stop if no training data available
            try:
                train_df = dataset.prepare("train", col_set="feature")
                if train_df.empty or len(train_df) < 10:
                    if verbose:
                        print(f"  → No training data available. Stopping walk-forward.")
                    break
            except Exception:
                if verbose:
                    print(f"  → Failed to prepare training data. Stopping walk-forward.")
                break
            
            # Train model
            model = ModelTrainer(seed=self.config.seed)
            try:
                model.train(dataset)
            except ValueError as e:
                if "Empty data" in str(e):
                    if verbose:
                        print(f"  → No training data available. Stopping walk-forward.")
                    break
                raise
            
            # Predict
            pred = model.predict(dataset)
            
            # Check if predictions are empty (no test data)
            if pred.empty:
                if verbose:
                    print(f"  → No test data available. Stopping walk-forward.")
                break
            
            # Apply smoothing
            pred_smooth = smooth_predictions(pred, halflife=self.config.smooth_window)
            
            self.predictions.append(pred_smooth)
            
            if verbose:
                print(f"  → Generated {len(pred_smooth)} predictions\n")
        
        # Concatenate all predictions
        all_preds = pd.concat(self.predictions)
        
        # Remove potential duplicates (overlapping windows)
        all_preds = all_preds[~all_preds.index.duplicated(keep='last')]
        
        # Sort by datetime
        all_preds = all_preds.sort_index()
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"  Walk-Forward Complete")
            print(f"  Total Predictions: {len(all_preds)}")
            print(f"{'='*60}\n")
        
        return all_preds


def run_walk_forward_backtest(
    test_start: str = '2023-01-01',
    test_end: str = None,
    topk: int = 4,
    config: WalkForwardConfig = None
) -> Tuple[pd.DataFrame, dict]:
    """
    Convenience function to run walk-forward backtest.
    
    Args:
        test_start: Start of test period
        test_end: End of test period (defaults to today)
        topk: Number of top ETFs to hold
        config: Walk-forward configuration
        
    Returns:
        Tuple of (backtest_report, positions)
    """
    from .backtest import BacktestEngine
    import qlib
    from .config import QLIB_PROVIDER_URI, QLIB_REGION
    
    # Initialize Qlib
    qlib.init(provider_uri=QLIB_PROVIDER_URI, region=QLIB_REGION, kernels=1)
    
    if test_end is None:
        test_end = datetime.datetime.now().strftime('%Y-%m-%d')
    
    # Run walk-forward validation
    validator = WalkForwardValidator(config=config)
    predictions = validator.run(test_start, test_end)
    
    # Run backtest
    engine = BacktestEngine(predictions)
    report, positions = engine.run(
        topk=topk,
        start_time=test_start,
        end_time=None
    )
    
    return report, positions
