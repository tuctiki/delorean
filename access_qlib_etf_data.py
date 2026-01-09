# data_acquisition.py

import qlib
from qlib.constant import REG_CN
from qlib.data import D
import pandas as pd

# 定义ETF池（可在后续步骤调整）
ETF_LIST = [
    "510300.SH",   # 沪深300ETF（主流宽基）
    "159339.SH",   # 中证A500ETF（您指定；若无数据, 可替换为"512050.SH"或"159360.SZ"等主流A500ETF）
    "512480.SH",   # 半导体ETF
    "516160.SH",   # 新能源车ETF
    "512690.SH",   # 白酒ETF
    "512800.SH",   # 银行ETF
    "512010.SH",   # 医药ETF
    "510630.SH",   # 消费ETF
    "515790.SH",   # 光伏ETF
    "512880.SH",   # 证券ETF（补充一个金融子行业）
]

def initialize_qlib(provider_uri="~/.qlib/qlib_data/cn_etf_data", region=REG_CN):
    """Initializes Qlib."""
    qlib.init(provider_uri=provider_uri, region=region)

def get_trading_calendar(start_time="2015-01-01", end_time="2026-01-01"):
    """Gets the trading calendar."""
    calendar = D.calendar(start_time=start_time, end_time=end_time)
    print("First 10 days of trading calendar:", calendar[:10])
    return calendar

def get_etf_data(etf_ticker, start_time="2015-01-01", end_time="2026-01-01"):
    """Gets historical data for a given ETF."""
    features = D.features([etf_ticker], ["$close", "$volume", "$adjclose"], start_time=start_time, end_time=end_time)
    print(f"\nRecent data sample for {etf_ticker}:")
    print(features.head(10))
    return features

def main():
    """Main function to acquire and display data."""
    start_time = "2015-01-01"
    end_time = "2026-01-01"

    initialize_qlib()
    initialize_qlib()
    get_trading_calendar(start_time=start_time, end_time=end_time)
    
    # Get data for all ETFs in the list
    for etf in ETF_LIST:
        get_etf_data(etf, start_time=start_time, end_time=end_time)

if __name__ == "__main__":
    main()
