from qlib.config import REG_CN

# ETF List
ETF_LIST = [
    # Broad Market (China)
    "510300.SH",    # 沪深300ETF
    "563360.SH",    # 华泰柏瑞中证A500ETF
    "159915.SZ",    # 新增：创业板ETF (ChiNext)
    "588000.SH",    # 新增：科创50ETF (STAR 50)
    "512100.SH",    # 新增：中证1000ETF (Small Cap)
    "510050.SH",    # 新增：上证50 (Large Cap Value)
    "510900.SH",    # 新增：H股ETF (H-Share)
    
    # Global / QDII (Available but segregated from main model)
    # "513050.SH",    # KWEB
    # "513100.SH",    # Nasdaq 100
    # "513500.SH",    # S&P 500
    # "159920.SZ",    # HSI
    
    # Commodities
    "518880.SH",    # 新增：黄金ETF (Gold)
    "512400.SH",    # 新增：有色金属 (Commodity Proxy)
    
    # Sector
    "512480.SH",    # 半导体
    "516160.SH",    # 新能源车
    "512690.SH",    # 白酒
    "512800.SH",    # 银行
    "512010.SH",    # 医药
    "510630.SH",    # 消费
    "515790.SH",    # 光伏
    "512880.SH",    # 证券
    "512660.SH",    # 新增：军工 (Defense)
    "510880.SH",    # 红利ETF (Defensive)
]

ETF_NAME_MAP = {
    # Broad Market (China)
    "510300.SH": "CSI 300",
    "563360.SH": "A500",
    "159915.SZ": "ChiNext (Startup)",
    "588000.SH": "STAR 50 (Tech)",
    "512100.SH": "CSI 1000 (Small Cap)",
    "510050.SH": "SSE 50 (Large Cap)",
    "510900.SH": "H-Share (HSCEI)",
    
    # Global / QDII
    "513050.SH": "KWEB (China Internet)",
    "513100.SH": "Nasdaq 100",
    "513500.SH": "S&P 500",
    "159920.SZ": "Hang Seng Index",
    
    # Commodities
    "518880.SH": "Gold",
    "512400.SH": "Non-Ferrous (Resources)",
    
    # Sector
    "512480.SH": "Semiconductor",
    "516160.SH": "New Energy",
    "512690.SH": "Spirit/Liquor",
    "512800.SH": "Bank",
    "512010.SH": "Pharma",
    "510630.SH": "Consumer",
    "515790.SH": "PV/Solar",
    "512880.SH": "Securities",
    "512660.SH": "Defense/Military",
    "510880.SH": "Dividend (RedChip)",
}

# Benchmark
BENCHMARK = "510300.SH"

# Time Range
START_TIME = "2015-01-01"
END_TIME = "2099-12-31" # Future-proof for live trading
TRAIN_END_TIME = "2022-12-31"
TEST_START_TIME = "2023-01-01"

# Qlib Config
QLIB_PROVIDER_URI = '~/.qlib/qlib_data/cn_etf_data'
QLIB_REGION = REG_CN

# MLflow Config
DEFAULT_EXPERIMENT_NAME = "ETF_Strategy"

# Default Backtest Parameters (used by Run Backtest button)
DEFAULT_BACKTEST_PARAMS = {
    "start_time": "2015-01-01",
    "train_end_time": "2022-12-31",
    "test_start_time": "2023-01-01",
    "topk": 4,
    "label_horizon": 1,
    "smooth_window": 10,
    "target_vol": 0.20, # Added default target vol
}

DEFAULT_TARGET_VOLATILITY = 0.20

# Live Trading Configuration (used by run_live_trading.py)
LIVE_TRADING_CONFIG = {
    "validation_days": 60,      # Days for out-of-sample validation
    "production_split_days": 14, # Days for production signal split
    "smooth_window": 10,        # EWMA halflife for signal smoothing (Reduced from 15)
    "buffer_size": 2,           # Hysteresis buffer for turnover control
    "label_horizon": 1,         # Forward return prediction horizon (Reduced from 5)
    "topk": 4,                  # Number of top ETFs to recommend (Reduced from 5)
}

# Output Config
import os
OUTPUT_DIR = "artifacts"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Model Hyperparameters
# Stage 1: Standard Qlib LGBModel (Experiment 8 Optimized)
# UPDATED 2026-01-15: Reduced regularization/constraints for robustness on small data
MODEL_PARAMS_STAGE1 = {
    "loss": "mse",
    "colsample_bytree": 0.6116,
    "learning_rate": 0.02,
    "subsample": 0.6,
    "lambda_l1": 0.1,  # Reduced from 0.5
    "lambda_l2": 0.1,  # Reduced from 0.5
    "max_depth": 5,
    "num_leaves": 19,
    "min_data_in_leaf": 20, # Reduced from 34
    "early_stopping_rounds": 100,
    "num_boost_round": 1000
    # Seed injected at runtime
}

# Stage 2: Refined Native LightGBM (Selected Features)
MODEL_PARAMS_STAGE2 = {
    "objective": "regression",
    "metric": "mse",
    "learning_rate": 0.03, # Slower learning for robustness
    "num_leaves": 15,      # Smaller trees
    "colsample_bytree": 0.6,
    "subsample": 0.6,
    "reg_alpha": 1.0,      # Stronger L1
    "reg_lambda": 1.0,     # Stronger L2
    "n_jobs": -1,
    "verbosity": -1,
    # Seed injected at runtime
}
