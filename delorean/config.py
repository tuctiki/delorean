from qlib.config import REG_CN

# ETF List
ETF_LIST = [
    "510300.SH",    # 沪深300ETF
    "563360.SH",    # 华泰柏瑞中证A500ETF
    "159915.SZ",    # 新增：创业板ETF (ChiNext)
    "588000.SH",    # 新增：科创50ETF (STAR 50)
    "512100.SH",    # 新增：中证1000ETF (Small Cap)
    "512480.SH",    # 半导体
    "516160.SH",    # 新能源车
    "512690.SH",    # 白酒
    "512800.SH",    # 银行
    "512010.SH",    # 医药
    "510630.SH",    # 消费
    "515790.SH",    # 光伏
    "512880.SH",    # 证券
    "510880.SH",    # 红利ETF (Defensive)
]

ETF_NAME_MAP = {
    "510300.SH": "CSI 300",
    "563360.SH": "A500",
    "159915.SZ": "ChiNext (Startup)",
    "588000.SH": "STAR 50 (Tech)",
    "512100.SH": "CSI 1000 (Small Cap)",
    "512480.SH": "Semiconductor",
    "516160.SH": "New Energy",
    "512690.SH": "Spirit/Liquor",
    "512800.SH": "Bank",
    "512010.SH": "Pharma",
    "510630.SH": "Consumer",
    "515790.SH": "PV/Solar",
    "512880.SH": "Securities",
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

# Output Config
import os
OUTPUT_DIR = "artifacts"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
