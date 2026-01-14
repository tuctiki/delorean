"""
Market regime detection utilities for ETF strategy.
Provides functions to determine bull/bear market status based on benchmark indicators.
"""

import pandas as pd
from typing import Tuple, Optional, Dict, Any
from datetime import datetime

from qlib.data import D
from .config import BENCHMARK, START_TIME, END_TIME, LIVE_TRADING_CONFIG


def calculate_regime_series(
    benchmark: str = BENCHMARK,
    start_time: str = START_TIME,
    end_time: str = END_TIME,
    ma_window: int = None
) -> pd.Series:
    """
    Calculate market regime as a time series.
    
    Uses moving average crossover to determine bull/bear periods.
    Bull = Close > MA, Bear = Close <= MA.
    
    Args:
        benchmark: Benchmark symbol to use for regime detection.
        start_time: Start date for calculation.
        end_time: End date for calculation.
        ma_window: Moving average window. Defaults to config value.
        
    Returns:
        Boolean series indexed by datetime (True=Bull, False=Bear).
    """
    if ma_window is None:
        ma_window = LIVE_TRADING_CONFIG.get("regime_ma_window", 60)
    
    # Fetch benchmark data
    bench_df = D.features(
        [benchmark], 
        ['$close'], 
        start_time=start_time, 
        end_time=end_time
    )
    
    if bench_df.empty:
        return pd.Series(dtype=bool)
    
    # Process index - drop instrument level
    bench_close = bench_df.droplevel(0) if bench_df.index.nlevels > 1 else bench_df
    bench_close.columns = ['close']
    
    # Calculate moving average
    bench_close['ma'] = bench_close['close'].rolling(window=ma_window).mean()
    
    # Define regime: True = Bull, False = Bear
    regime = bench_close['close'] > bench_close['ma']
    regime.name = 'regime'
    
    return regime


def get_current_regime(
    date: datetime = None,
    benchmark: str = BENCHMARK
) -> Tuple[bool, Dict[str, Any]]:
    """
    Get market regime status for a specific date.
    
    Args:
        date: Date to check. Defaults to latest available.
        benchmark: Benchmark symbol to use.
        
    Returns:
        Tuple of (is_bull: bool, market_data: dict with close/ma values).
    """
    ma_window = LIVE_TRADING_CONFIG.get("regime_ma_window", 60)
    
    if date is None:
        date = datetime.now()
    
    date_str = date.strftime("%Y-%m-%d") if hasattr(date, 'strftime') else str(date)
    
    # Fetch current close and MA
    fields = ['$close', f'Mean($close, {ma_window})']
    feat_df = D.features([benchmark], fields, start_time=date_str, end_time=date_str)
    
    market_data = {
        "benchmark_close": 0.0,
        "benchmark_ma": 0.0,
        "ma_window": ma_window
    }
    
    is_bull = True  # Default to bull if data unavailable
    
    if not feat_df.empty:
        market_data["benchmark_close"] = float(feat_df.iloc[0, 0])
        market_data["benchmark_ma"] = float(feat_df.iloc[0, 1])
        
        is_bull = market_data["benchmark_close"] > market_data["benchmark_ma"]
    
    return is_bull, market_data


def get_regime_status_string(is_bull: bool) -> str:
    """
    Get human-readable regime status.
    
    Args:
        is_bull: Boolean indicating bull market.
        
    Returns:
        "Bull" or "Bear" string.
    """
    return "Bull" if is_bull else "Bear"
