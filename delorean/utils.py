"""
Delorean Utilities Module

Centralized utility functions for common operations across the codebase.
"""
import pandas as pd
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def smooth_predictions(
    pred: pd.Series, 
    halflife: int = 15, 
    level_name: str = 'instrument'
) -> pd.Series:
    """
    Apply EWMA smoothing to prediction scores.
    
    Smooths predictions per instrument to reduce noise and improve signal stability.
    This is particularly useful for reducing turnover in trading strategies.
    
    Args:
        pred: Prediction series with MultiIndex (datetime, instrument).
        halflife: EWMA halflife in days (default: 15).
        level_name: Name of the instrument level in the index (default: 'instrument').
        
    Returns:
        Smoothed prediction series with same index structure.
        
    Example:
        >>> pred_smooth = smooth_predictions(pred_raw, halflife=10)
    """
    if pred.empty:
        logger.warning("Empty prediction series provided to smooth_predictions")
        return pred
    
    # Detect the correct level name
    if pred.index.names[1] == 'instrument':
        level_name = 'instrument'
    elif len(pred.index.names) > 1:
        level_name = pred.index.names[1]
    
    # Apply EWMA per instrument
    pred_smooth = pred.groupby(level=level_name).apply(
        lambda x: x.ewm(halflife=halflife, min_periods=1).mean()
    )
    
    # Clean up the index structure
    if pred_smooth.index.nlevels > 2:
        pred_smooth = pred_smooth.droplevel(0)
    
    # Ensure datetime is first level
    if pred_smooth.index.names[0] != 'datetime' and 'datetime' in pred_smooth.index.names:
        pred_smooth = pred_smooth.swaplevel()
    
    pred_smooth = pred_smooth.dropna().sort_index()
    
    logger.debug(f"Smoothed predictions with halflife={halflife}, shape={pred_smooth.shape}")
    return pred_smooth


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
