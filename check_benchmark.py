
import qlib
from qlib.data import D
import pandas as pd
from constants import QLIB_PROVIDER_URI, QLIB_REGION

qlib.init(provider_uri=QLIB_PROVIDER_URI, region=QLIB_REGION)

def calculate_benchmark_metrics():
    start_date = "2023-01-01"
    end_date = "2025-12-31" # Data might only go up to today, but request was for "past 3 years" or available.
    # Actually data goes up to 2099? No, data we have.
    # Let's check updated data.
    
    df = D.features(["510300.SH"], ["$close"], start_time=start_date, end_time=end_date)
    df = df.droplevel(0) # Drop instrument index
    df.columns = ["close"]
    
    # Calculate daily returns
    df["return"] = df["close"].pct_change()
    
    # Cumulative Return
    df["cumulative"] = (1 + df["return"]).cumprod()
    
    # Metrics
    total_ret = df["cumulative"].iloc[-1] - 1
    days = (df.index[-1] - df.index[0]).days
    annualized_ret = (1 + total_ret) ** (365 / days) - 1
    
    # Max Drawdown
    rolling_max = df["cumulative"].cummax()
    drawdowns = df["cumulative"] / rolling_max - 1
    max_dd = drawdowns.min()
    
    print(f"Benchmark (HS300) 2023-2025:")
    print(f"Total Return: {total_ret*100:.2f}%")
    print(f"Annualized Return: {annualized_ret*100:.2f}%")
    print(f"Max Drawdown: {max_dd*100:.2f}%")

if __name__ == "__main__":
    calculate_benchmark_metrics()
