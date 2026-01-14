"""
Delorean Utilities Module

Centralized utility functions for common operations across the codebase.
"""
import pandas as pd
import logging
import random
import numpy as np
from typing import Optional

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

