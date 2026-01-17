
import sys
import os
import pandas as pd
import numpy as np
import qlib
from qlib.data import D
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from delorean.config import QLIB_PROVIDER_URI, QLIB_REGION, ETF_LIST
from delorean.data import ETFDataHandler

# Reuse audit logic
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from audit_factors_enhanced import evaluate_single_factor, calculate_correlation_matrix

def init_qlib():
    provider_uri = os.path.expanduser(QLIB_PROVIDER_URI)
    qlib.init(provider_uri=provider_uri, region=QLIB_REGION)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="artifacts/mined_alphas.txt")
    parser.add_argument("--output", default="artifacts/mined_alphas_audit.csv")
    args = parser.parse_args()
    
    init_qlib()
    
    # Load mined alphas
    with open(args.input, "r") as f:
        alphas = [line.strip() for line in f if line.strip()]
        
    print(f"Loaded {len(alphas)} mined alphas.")
    
    # Load Existing Factors (for correlation check)
    base_exprs, base_names = ETFDataHandler.get_custom_factors()
    
    # Validation Period (OOS)
    START = "2023-01-01"
    END = "2025-12-31"
    LABEL_HORIZON = 5
    
    print(f"Validating on OOS Period: {START} to {END}")
    
    # Prepare Qlib Fields
    # Names for mined alphas: Alpha_Gen_1, Alpha_Gen_2...
    mined_names = [f"Alpha_Gen_{i+1}" for i in range(len(alphas))]
    
    all_exprs = base_exprs + alphas + [f"Ref($close, -{LABEL_HORIZON}) / $close - 1"]
    all_names = base_names + mined_names + ["label"]
    
    # Create a mapping to handle potential duplicates or errors
    # Actually Qlib might error if expressions are identical?
    # We'll rely on D.features
    
    try:
        df = D.features(ETF_LIST, all_exprs, start_time=START, end_time=END, freq='day')
        df.columns = all_names
        df = df.dropna()
    except Exception as e:
        print(f"Error loading data (possibly invalid formula): {e}")
        return

    print(f"Loaded {len(df)} rows.")
    
    results = []
    
    # Evaluate Mined Factors
    for i, name in enumerate(mined_names):
        print(f"Auditing {name}...")
        metrics = evaluate_single_factor(df[name], df["label"], name)
        if metrics:
            metrics["Formula"] = alphas[i]
            results.append({"Name": name, **metrics})
            
    if not results:
        print("No valid factors found.")
        return
        
    df_res = pd.DataFrame(results)
    
    # Correlation Check against BASELINE
    # We want factors that are NOT correlated with existing ones
    print("Checking Correlations...")
    df_all_factors = df[base_names + mined_names]
    corr_matrix = calculate_correlation_matrix(df_all_factors)
    
    final_candidates = []
    
    for _, row in df_res.iterrows():
        name = row["Name"]
        # Check max correlation with BASE factors
        corr_with_base = corr_matrix.loc[name, base_names].abs().max()
        best_match = corr_matrix.loc[name, base_names].abs().idxmax()
        
        row["Max_Corr_Base"] = corr_with_base
        row["Corr_Partner"] = best_match
        
        # Criteria: IC > 0.03, Correlation < 0.7
        if abs(row["IC"]) > 0.02 and corr_with_base < 0.7:
             row["Status"] = "PASS"
        else:
             row["Status"] = "FAIL"
             
        final_candidates.append(row)
        
    df_final = pd.DataFrame(final_candidates)
    df_final = df_final.sort_values("IC", key=abs, ascending=False)
    
    print("\n" + "="*60)
    print("VALIDATION RESULTS")
    print("="*60)
    print(df_final[["Name", "IC", "ICIR", "Max_Corr_Base", "Status"]].to_string(index=False))
    
    df_final.to_csv(args.output, index=False)
    print(f"\nSaved to {args.output}")

if __name__ == "__main__":
    main()
