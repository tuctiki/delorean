import akshare as ak
import pandas as pd
import os
import re

# From data_acquisition.py, copied here to avoid circular dependency
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

def get_etf_hist_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetches historical daily ETF data using AkShare.

    Args:
        symbol (str): ETF symbol (e.g., "510300.SH").
        start_date (str): Start date in "YYYYMMDD" format.
        end_date (str): End date in "YYYYMMDD" format.

    Returns:
        pd.DataFrame: DataFrame containing historical ETF data.
    """
    print(f"Fetching ETF data for {symbol} from {start_date} to {end_date}...")
    try:
        # Extract the numerical part of the symbol, e.g., "510300" from "510300.SH"
        etf_code = re.match(r"(\d+)", symbol).group(1)
        
        df = ak.fund_etf_hist_em(
            symbol=etf_code,
            period="daily",
            start_date=start_date,
            end_date=end_date,
            adjust="qfq"  # Historical forward-adjusted
        )
        if df.empty:
            print(f"No data fetched for {symbol}")
        return df
    except Exception as e:
        print(f"Error fetching ETF data for {symbol}: {e}")
        return pd.DataFrame()

def format_for_qlib(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Formats the AkShare DataFrame into Qlib's expected format.

    Args:
        df (pd.DataFrame): DataFrame from AkShare with ETF data.
        symbol (str): ETF symbol.

    Returns:
        pd.DataFrame: Formatted DataFrame for Qlib.
    """
    if df.empty:
        return pd.DataFrame()

    # Rename columns to Qlib's expected format
    # Columns from ak.fund_etf_hist_em: '日期', '开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌幅', '涨跌额', '换手率'
    required_akshare_cols = {
        "日期": "date",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
        # "前复权因子": "factor" # Not directly available for ETF in ak.fund_etf_hist_em output
    }
    
    available_akshare_cols = {
        ak_col: qlib_col for ak_col, qlib_col in required_akshare_cols.items() 
        if ak_col in df.columns
    }

    df_qlib = df.rename(columns=available_akshare_cols)

    # Select and reorder necessary columns. Qlib requires: date, open, close, high, low, volume, (factor)
    target_qlib_cols = ["date", "open", "close", "high", "low", "volume"]
    
    # Fill missing target columns with NaN
    for col in target_qlib_cols:
        if col not in df_qlib.columns:
            df_qlib[col] = pd.NA

    df_qlib = df_qlib[target_qlib_cols]

    # Convert 'date' column to datetime and set as index
    df_qlib["date"] = pd.to_datetime(df_qlib["date"])
    df_qlib = df_qlib.set_index("date")

    # Add 'symbol' column
    df_qlib["symbol"] = symbol

    return df_qlib

def main():
    start_date = "20150101" # YYYYMMDD format
    end_date = "20260101"   # YYYYMMDD format

    # Define output directory for ETF data
    output_dir = os.path.expanduser("~/.qlib/csv_data/akshare_etf_data")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    for symbol in ETF_LIST:
        etf_data = get_etf_hist_data(symbol, start_date, end_date)
        if not etf_data.empty:
            qlib_formatted_data = format_for_qlib(etf_data, symbol)
            if not qlib_formatted_data.empty:
                output_path = os.path.join(output_dir, f"{symbol}.csv")
                qlib_formatted_data.to_csv(output_path)
                print(f"Formatted data for {symbol} saved to: {output_path}")
            else:
                print(f"No formatted data for {symbol}")
        else:
            print(f"Failed to fetch data for {symbol}")

if __name__ == "__main__":
    main()