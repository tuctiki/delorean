import akshare as ak
import pandas as pd

candidates = ["513520", "164824"]

print(f"{'Code':<10} {'Result':<20} {'Rows':<10}")
print("-" * 40)

for code in candidates:
    try:
        # standard etf interface
        df = ak.fund_etf_hist_em(symbol=code, period="daily", start_date="20230101", end_date="20230110", adjust="qfq")
        if not df.empty:
            print(f"{code:<10} {'Success':<20} {len(df):<10}")
        else:
            print(f"{code:<10} {'Empty':<20} {0:<10}")
    except Exception as e:
        print(f"{code:<10} {'Failed':<20} {str(e)}")
