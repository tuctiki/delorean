from qlib.config import REG_CN

# ETF List
ETF_LIST = [
    "510300.SH",    # 沪深300ETF
    "563360.SH",    # 新增：华泰柏瑞中证A500ETF（2026年规模最大，推荐首选）
    # 若Yahoo无563360数据，可换 "512050.SH" (华夏) 或 "159352.SH" (南方)
    "512480.SH",    # 半导体
    "516160.SH",    # 新能源车
    "512690.SH",    # 白酒
    "512800.SH",    # 银行
    "512010.SH",    # 医药
    "510630.SH",    # 消费
    "515790.SH",    # 光伏
    "512880.SH",    # 证券
]

# Benchmark
BENCHMARK = "510300.SH"

# Time Range
START_TIME = "2015-01-01"
END_TIME = "2025-12-25"
TRAIN_END_TIME = "2023-12-31"
TEST_START_TIME = "2024-01-01"

# Qlib Config
QLIB_PROVIDER_URI = '~/.qlib/qlib_data/cn_etf_data'
QLIB_REGION = REG_CN

# Output Config
import os
OUTPUT_DIR = "artifacts"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
