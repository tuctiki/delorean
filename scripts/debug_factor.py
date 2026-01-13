
import sys
import os
import pandas as pd
import numpy as np

# Add project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from delorean.data import ETFDataLoader
from delorean.config import ETF_LIST, QLIB_PROVIDER_URI
import qlib

def debug():
    # Force single thread to debug error
    qlib.init(provider_uri=QLIB_PROVIDER_URI, region="cn", kernels=1)
    
    print("Loading data with new factor...")
    # Load just a small chunk to test
    loader = ETFDataLoader(
        start_time="2020-01-01", 
        end_time="2020-12-31"
    )
    
    # This triggers the handler init 
    loader.load_data()
    
    # Now try to get the dataframe
    df = loader.dataset.prepare("train", col_set=["feature"])
    
    print("Data loaded successfully.")
    print("Shape:", df.shape)
    
    # Check if our new factor is there
    # It should be the last column usually, or we search for it
    # The columns are MultiIndex (feature, name)
    
    # Get custom names from handler
    _, names = loader.handler.get_custom_factors()
    new_factor_name = "Alpha_Miner_01"
    
    if new_factor_name in names:
        print(f"Factor '{new_factor_name}' found.")
        
        # Extract the column
        # The columns are likely just strings if flattened? 
        # Or (feature, name) tuples. Qlib DatasetH usually returns (feature, name) columns
        
        # Let's just print columns to be sure
        # print(df.columns)
        
        try:
            # Check for NaNs/Infs
            # We need to find the column index corresponding to it
            # The df columns are MultiIndex: [('feature', 'MarketCap_Liquidity'), ...]
            
            target_col = ('feature', new_factor_name)
            if target_col in df.columns:
                series = df[target_col]
                print(f"Stats for {new_factor_name}:")
                print("Mean:", series.mean())
                print("Min:", series.min())
                print("Max:", series.max())
                print("NaNs:", series.isna().sum())
                print("Infs:", np.isinf(series).sum())
            else:
                print(f"Column {target_col} not found in DataFrame.")
                print("Columns:", df.columns[:5])
                
        except Exception as e:
            print(f"Error analyzing column: {e}")
            
    else:
        print(f"Factor '{new_factor_name}' NOT found in custom_names.")

if __name__ == "__main__":
    debug()
