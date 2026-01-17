
import pandas as pd
import qlib
from qlib.data import D
import sys
import os

# Setup path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from delorean.conf import QLIB_PROVIDER_URI, QLIB_REGION, ETF_LIST
from delorean.alphas.factors import get_production_factors

def audit_yearly():
    qlib.init(provider_uri=QLIB_PROVIDER_URI, region=QLIB_REGION)
    
    expressions, names = get_production_factors()
    # Focus on Alpha_Gen_8 but run all
    fields = list(expressions) + ["Ref($close, -1) / $close - 1"]
    names_with_label = list(names) + ["label"]
    
    years = range(2020, 2027) # 2020 to 2026 inclusive
    
    print(f"{'Year':<6} | {'Factor':<20} | {'Rank IC':<10}")
    print("-" * 45)
    
    for year in years:
        start = f"{year}-01-01"
        end = f"{year}-12-31"
        
        try:
            df = D.features(ETF_LIST, fields, start_time=start, end_time=end)
            if df.empty:
                continue
                
            df.columns = names_with_label
            df = df.dropna()
            
            for name in names:
                ic = df[name].corr(df["label"], method="spearman")
                print(f"{year:<6} | {name:<20} | {ic:.4f}")
            print("-" * 45)
            
        except Exception as e:
            print(f"Error for {year}: {e}")

if __name__ == "__main__":
    audit_yearly()
