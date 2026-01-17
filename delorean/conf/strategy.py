
# Default Backtest Parameters (used by Run Backtest button)
DEFAULT_BACKTEST_PARAMS = {
    "start_time": "2015-01-01",
    "train_end_time": "2022-12-31",
    "test_start_time": "2023-01-01",
    "topk": 4,
    "label_horizon": 1,
    "smooth_window": 10,
    "target_vol": 0.20,
    "signal_halflife": 5,        # EMA smoothing for turnover reduction
    "buffer": 3,                  # Increased from 2 for wider hysteresis
    "rebalance_threshold": 0.05,  # Increased from 0.02 (5% threshold)
}

DEFAULT_TARGET_VOLATILITY = 0.20

# Live Trading Configuration (used by run_live_trading.py)
LIVE_TRADING_CONFIG = {
    "validation_days": 60,       # Days for out-of-sample validation
    "production_split_days": 14, # Days for production signal split
    "smooth_window": 10,         # EWMA halflife for signal smoothing (Reduced from 15)
    "buffer_size": 3,            # Hysteresis buffer for turnover control (Increased from 2)
    "label_horizon": 1,          # Forward return prediction horizon (Reduced from 5)
    "topk": 4,                   # Number of top ETFs to recommend (Reduced from 5)
    "target_vol": 0.20,          # Default target volatility
    "signal_halflife": 5,        # EMA smoothing for prediction scores
    "rebalance_threshold": 0.05, # Rebalancing threshold (5%)
}
