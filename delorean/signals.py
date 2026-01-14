"""
Signal processing utilities for ETF strategy.
Provides functions for signal smoothing and index manipulation.
"""

import pandas as pd
from typing import Optional


def smooth_predictions(
    pred: pd.Series, 
    halflife: int = 10, 
    min_periods: int = 1
) -> pd.Series:
    """
    Apply EWMA smoothing to prediction scores.
    
    Smooths predictions per instrument using exponentially weighted moving average
    to reduce noise and turnover.
    
    Args:
        pred: Prediction series with MultiIndex (datetime, instrument).
        halflife: EWMA halflife in days. Higher = smoother, lower turnover.
        min_periods: Minimum observations required for valid output.
        
    Returns:
        Smoothed prediction series with clean (datetime, instrument) index.
    """
    if pred.empty:
        return pred
    
    # Identify the instrument level name
    if pred.index.names[1] == 'instrument':
        level_name = 'instrument'
    else:
        level_name = pred.index.names[1]
    
    # Apply EWMA per instrument
    smoothed = pred.groupby(level=level_name).apply(
        lambda x: x.ewm(halflife=halflife, min_periods=min_periods).mean()
    )
    
    # Clean up index (groupby may add extra level)
    smoothed = clean_prediction_index(smoothed)
    
    return smoothed


def clean_prediction_index(pred: pd.Series) -> pd.Series:
    """
    Clean and normalize prediction index to (datetime, instrument) format.
    
    Handles various index issues that can occur after groupby operations:
    - Removes redundant levels
    - Ensures datetime is first level
    - Sorts by datetime
    
    Args:
        pred: Prediction series with potentially messy MultiIndex.
        
    Returns:
        Cleaned prediction series with (datetime, instrument) index.
    """
    if pred.empty:
        return pred
    
    # Remove redundant level if groupby added one
    if pred.index.nlevels > 2:
        pred = pred.droplevel(0)
    
    # Ensure (datetime, instrument) order
    if pred.index.names[0] != 'datetime' and 'datetime' in pred.index.names:
        pred = pred.swaplevel()
    
    # Drop NaN and sort
    pred = pred.dropna().sort_index()
    
    return pred


def rank_predictions(pred: pd.Series) -> pd.Series:
    """
    Convert prediction scores to cross-sectional ranks.
    
    Ranks instruments within each datetime, which can be useful
    for strategies that only care about relative ordering.
    
    Args:
        pred: Prediction series with MultiIndex (datetime, instrument).
        
    Returns:
        Rank series (1 = best) with same index.
    """
    if pred.empty:
        return pred
    
    return pred.groupby(level='datetime').rank(ascending=False)
