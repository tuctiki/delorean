"""
Delorean Utilities Module

Centralized utility functions for common operations across the codebase.
"""
import pandas as pd
import logging
import random
import numpy as np
from typing import Optional, Union, List, Tuple

logger = logging.getLogger(__name__)

# Re-export smooth_predictions from signals module for backward compatibility
from delorean.signals import smooth_predictions


def fix_seed(seed: int = 42) -> None:
    """
    Fix random seeds for reproducibility across all libraries.
    
    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    
    logger.debug(f"Random seed fixed: {seed}")


def calculate_rank_ic(
    pred: pd.Series,
    labels: pd.Series,
    method: str = 'spearman'
) -> float:
    """
    Calculate Rank Information Coefficient (IC) between predictions and labels.
    
    The Rank IC measures the correlation between predicted rankings and actual
    forward returns, which is a key metric for evaluating factor quality.
    
    Args:
        pred: Prediction series.
        labels: Label series (typically forward returns).
        method: Correlation method ('spearman' or 'pearson').
        
    Returns:
        Rank IC value between -1 and 1.
        
    Example:
        >>> ic = calculate_rank_ic(predictions, actual_returns)
    """
    # Align indices
    common_idx = pred.index.intersection(labels.index)
    
    if common_idx.empty:
        logger.warning("No common indices between predictions and labels")
        return 0.0
    
    pred_aligned = pred.loc[common_idx]
    labels_aligned = labels.loc[common_idx]
    
    # Handle multi-column labels (take first column)
    if hasattr(labels_aligned, 'iloc') and hasattr(labels_aligned, 'ndim'):
        if labels_aligned.ndim > 1:
            labels_aligned = labels_aligned.iloc[:, 0]
    
    ic = pred_aligned.corr(labels_aligned, method=method)
    return ic


def get_benchmark_ma_ratio(
    benchmark: str = "510300.SH",
    window: int = 60,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None
) -> pd.Series:
    """
    Calculate the benchmark price to moving average ratio.
    
    This ratio is used for market regime detection:
    - Ratio > 1.0: Bull market (price above MA)
    - Ratio < 1.0: Bear market (price below MA)
    
    Args:
        benchmark: Benchmark instrument code.
        window: Moving average window in days (default: 60).
        start_time: Start date for data fetch.
        end_time: End date for data fetch.
        
    Returns:
        Series of price/MA ratio indexed by datetime.
    """
    from qlib.data import D
    
    expr = f'$close / Mean($close, {window})'
    
    try:
        df = D.features([benchmark], [expr], start_time=start_time, end_time=end_time)
        if df.empty:
            logger.warning(f"No data returned for benchmark {benchmark}")
            return pd.Series(dtype=float)
        
        # Return as single-level series (datetime only)
        ratio = df.iloc[:, 0]
        if ratio.index.nlevels > 1:
            ratio = ratio.droplevel('instrument')
        
        return ratio
    except Exception as e:
        logger.error(f"Failed to fetch benchmark MA ratio: {e}")
        return pd.Series(dtype=float)


def fetch_regime_ratio(
    benchmark: str,
    start_time: Union[str, pd.Timestamp],
    end_time: Union[str, pd.Timestamp]
) -> pd.Series:
    """
    Fetch market regime ratio (Price/MA60) for the benchmark.
    
    This is the primary regime detection signal used by the strategy:
    - Ratio >= 1.0: Bull market (price above 60-day MA)
    - Ratio < 1.0: Bear market (price below 60-day MA)
    
    Args:
        benchmark: Benchmark instrument code (e.g., "510300.SH").
        start_time: Start date for data fetch.
        end_time: End date for data fetch.
        
    Returns:
        pd.Series indexed by datetime with regime ratio values.
        Returns empty Series on error.
    """
    from qlib.data import D
    
    try:
        fields = ['$close / Mean($close, 60)']
        feat_df = D.features([benchmark], fields, start_time=start_time, end_time=end_time)
        
        if feat_df.empty:
            logger.warning(f"No regime data returned for {benchmark}")
            return pd.Series(dtype=float)
        
        # Extract Series and drop instrument level (single benchmark)
        regime_series = feat_df.iloc[:, 0]
        if regime_series.index.nlevels > 1:
            regime_series = regime_series.droplevel('instrument')
        
        return regime_series
    except Exception as e:
        logger.error(f"Failed to fetch regime ratio for {benchmark}: {e}")
        return pd.Series(dtype=float)


def fetch_volatility_feature(
    instruments: Union[str, List[str]],
    start_time: Union[str, pd.Timestamp],
    end_time: Union[str, pd.Timestamp]
) -> pd.Series:
    """
    Fetch 20-day rolling volatility for given instruments.
    
    This is used for Risk Parity weighting and Target Volatility scaling.
    
    Args:
        instruments: Single instrument code or list of codes.
        start_time: Start date for data fetch.
        end_time: End date for data fetch.
        
    Returns:
        pd.Series indexed by (datetime, instrument) with VOL20 values.
        Returns empty Series on error.
    """
    from qlib.data import D
    
    # Normalize to list
    if isinstance(instruments, str):
        instruments = [instruments]
    
    try:
        fields = ['Std($close/Ref($close,1)-1, 20)']
        vol_df = D.features(instruments, fields, start_time=start_time, end_time=end_time)
        
        if vol_df.empty:
            logger.warning(f"No volatility data returned for instruments")
            return pd.Series(dtype=float)
        
        # Return as Series with (datetime, instrument) index
        vol_series = vol_df.iloc[:, 0]
        # Qlib returns (instrument, datetime), so we swap levels
        if vol_series.index.names == ['instrument', 'datetime']:
             vol_series = vol_series.swaplevel().sort_index()
        return vol_series
    except Exception as e:
        logger.error(f"Failed to fetch volatility feature: {e}")
        return pd.Series(dtype=float)


def run_standard_backtest(
    pred: pd.Series,
    topk: int,
    buffer: int = 2,
    target_vol: float = 0.20,
    use_regime_filter: bool = True,
    use_trend_filter: bool = False,
    n_drop: int = 1,
    rebalance_threshold: float = 0.05,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None
) -> Tuple[pd.DataFrame, dict]:
    """
    Run backtest with standard strategy parameters.
    
    This is a convenience wrapper around BacktestEngine that applies
    the canonical strategy configuration used across the codebase.
    
    Args:
        pred: Prediction scores indexed by (datetime, instrument).
        topk: Number of stocks to hold.
        buffer: Rank buffer for hysteresis (default: 2).
        target_vol: Annualized target volatility (default: 0.20).
        use_regime_filter: Enable market regime filter (default: True).
        use_trend_filter: Enable per-asset trend filter (default: False).
        start_time: Backtest start time (optional).
        end_time: Backtest end time (optional).
        
    Returns:
        Tuple of (report DataFrame, positions dict).
    """
    from delorean.backtest import BacktestEngine
    
    engine = BacktestEngine(pred)
    report, positions = engine.run(
        topk=topk,
        buffer=buffer,
        target_vol=target_vol,
        use_regime_filter=use_regime_filter,

        use_trend_filter=use_trend_filter,
        n_drop=n_drop,
        rebalance_threshold=rebalance_threshold,
        start_time=start_time,
        end_time=end_time
    )
    
    return report, positions
