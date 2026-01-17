
import pandas as pd
import qlib
from qlib.data import D
from qlib.contrib.evaluate import risk_analysis
import sys
import os
import numpy as np

# Setup path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from delorean.conf import QLIB_PROVIDER_URI, QLIB_REGION, ETF_LIST, START_TIME, END_TIME, TEST_START_TIME
from delorean.alphas.factors import get_production_factors

def audit_factors(start_date, end_date):
    """
    Audit current production factors:
    1. Calculate IC (Information Coefficient) for each factor on Test Data.
    2. Check correlation between factors.
    """
    
    # 1. Initialize Qlib
    qlib.init(provider_uri=QLIB_PROVIDER_URI, region=QLIB_REGION)
    
    print(f"Auditing Factors on Period: {start_date} to {end_date}")
    
    # 2. Get Factors
    expressions, names = get_production_factors()
    
    # 3. Prepare Data
    # Fetch factor values and labels
    # Label: Forward 1-day return
    fields = list(expressions) + ["Ref($close, -1) / $close - 1"]
    names_with_label = list(names) + ["label"]
    
    print("Fetching data...")
    # Fetching for ALL ETFs in the universe
    df = D.features(ETF_LIST, fields, start_time=start_date, end_time=end_date)
    df.columns = names_with_label
    df = df.dropna()
    
    if df.empty:
        print("No data found for the test period!")
        return

    # 4. Calculate IC for each factor
    print("\n--- Factor Performance (Rank IC) ---")
    performance = []
    
    for name in names:
        # Rank IC: Spearman correlation between Factor and Label
        ic = df[name].corr(df["label"], method="spearman")
        print(f"{name:<20}: IC = {ic:.4f}")
        performance.append({"Factor": name, "Rank_IC": ic})
        
    perf_df = pd.DataFrame(performance)
    
    # 5. Correlation Matrix
    print("\n--- Factor Correlation Matrix ---")
    factor_df = df[list(names)]
    # Rank correlation usually better for factors
    corr_matrix = factor_df.rank().corr()
    print(corr_matrix)
    
    # 6. Check for "Negative IC" warnings
    print("\n--- Audit Warnings ---")
    for _, row in perf_df.iterrows():
        if row["Rank_IC"] < 0:
            print(f"[WARNING] {row['Factor']} has NEGATIVE IC ({row['Rank_IC']:.4f}). Consider flipping sign or removing.")
        elif row["Rank_IC"] < 0.02:
            print(f"[NOTE] {row['Factor']} has WEAK IC ({row['Rank_IC']:.4f}).")
            
    # 7. Check for high correlation
    print("\n--- Redundancy Check ---")
    high_corr_pairs = []
    headers = corr_matrix.columns
    for i in range(len(headers)):
        for j in range(i+1, len(headers)):
            c = corr_matrix.iloc[i, j]
            if abs(c) > 0.7:
                pair = f"{headers[i]} <-> {headers[j]}"
                print(f"[HIGH CORR] {pair} : {c:.4f}")
                
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Audit Alpha Factors")
    parser.add_argument("--start", type=str, default=TEST_START_TIME, help="Start Date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=END_TIME, help="End Date (YYYY-MM-DD)")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    audit_factors(start_date=args.start, end_date=args.end)
