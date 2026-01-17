import os
from qlib.config import REG_CN

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

# Output Config
OUTPUT_DIR = "artifacts"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
