import os
import sys
import pandas as pd
import numpy as np
import logging
from gplearn.genetic import SymbolicTransformer
from sklearn.model_selection import train_test_split

# Add project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from delorean.data import ETFDataLoader
from delorean.config import ETF_LIST, START_TIME, END_TIME, BENCHMARK, QLIB_PROVIDER_URI

import qlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("delorean.miner")

def main():
    qlib.init(provider_uri=QLIB_PROVIDER_URI, region="cn")
    
    # 1. Load Data
    # For GP, we need flat X and y.
    # We load "train" segment.
    TRAIN_END = "2021-12-31" 
    
    logger.info("Loading Data for Mining...")
    loader = ETFDataLoader(ETF_LIST, start_time=START_TIME, end_time=TRAIN_END)
    dataset = loader.load_data(train_end=TRAIN_END)
    
    # 2. Prepare X, y for gplearn
    # Qlib DatasetH.prepare returns dataframe
    df_train = dataset.prepare("train", col_set=["feature", "label"])
    
    logger.info(f"Shape before cleaning: {df_train.shape}")
    
    # 1. Drop columns that are mostly NaN (e.g., VWAP0 if missing)
    # Threshold: Drop cols with > 20% NaNs
    limit = len(df_train) * 0.2
    nan_counts = df_train.isna().sum()
    bad_cols = nan_counts[nan_counts > limit].index
    if not bad_cols.empty:
        logger.warning(f"Dropping columns with >20% NaNs: {list(bad_cols)}")
        df_train = df_train.drop(columns=bad_cols)
    
    # 2. Drop Remaining Rows
    df_train = df_train.dropna()
    logger.info(f"Shape after cleaning: {df_train.shape}")
    
    if df_train.empty:
        logger.error("Training data is empty after dropping NaNs! Aborting.")
        return
    
    X = df_train["feature"].values
    y = df_train["label"].values.ravel() # Flatten label
    
    feature_names = list(df_train["feature"].columns)
    
    logger.info(f"Data Loaded: {X.shape}, Features: {len(feature_names)}")
    
    # 3. Configure GP
    # We want to find NEW features that correlate with Y.
    # SymbolicTransformer generates new features.
    
    function_set = ['add', 'sub', 'mul', 'div', 'sqrt', 'log', 'abs', 'neg', 'inv']
    
    logger.info("Starting Genetic Programming Evolution...")
    est = SymbolicTransformer(
        generations=10,        # Quick run for MVP
        population_size=1000,
        hall_of_fame=50,
        n_components=10,
        function_set=function_set,
        parsimony_coefficient=0.001,
        max_samples=0.9,
        verbose=1,
        random_state=42,
        n_jobs=-1
    )
    
    est.fit(X, y)
    
    # 4. Extract Top Programs
    logger.info("Evolution Complete. Top Discovered Programs:")
    
    for i, program in enumerate(est):
        # program is a _Program object
        # We can print it. It uses X0, X1 etc.
        # We need to map X0 -> feature_name
        
        # Simple string replacement (naive but works for MVP display)
        # Note: gplearn str() representation uses X0, X1...
        
        raw_expr = str(program)
        
        # Replace X{i} with real names
        # We must replace in reverse order (X10 before X1) to avoid partial matches
        for idx in range(len(feature_names)-1, -1, -1):
            raw_expr = raw_expr.replace(f"X{idx}", feature_names[idx])
            
        print(f"\n[Factor {i+1}]")
        print(f"Structure: {raw_expr}")
        # Note: This structure is somewhat readable but needs conversion to Qlib expression syntax
        # e.g., 'add(X0, X1)' -> '$open + $close'
        # This is a complex translation task for later.
        
        # Analyze fitness? SymbolicTransformer doesn't expose fitness easily in iteration
        # but we know they are the top components.

if __name__ == "__main__":
    main()
