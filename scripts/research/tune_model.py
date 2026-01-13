import optuna
import os
import sys
import pandas as pd
import numpy as np
import logging

# Add project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from delorean.data import ETFDataLoader
from delorean.model import ModelTrainer
from delorean.config import ETF_LIST, START_TIME, END_TIME, BENCHMARK, QLIB_PROVIDER_URI

import qlib
from qlib.data import D

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("delorean.tuner")

def objective(trial):
    # 1. Suggest Hyperparameters
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 63),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "lambda_l1": trial.suggest_float("lambda_l1", 0.0, 10.0),
        "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 10.0),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 100),
        "verbosity": -1
    }
    
    # 2. Data Preparation (Cached to avoid reloading every trial?)
    # For simplicity, we assume Qlib handles caching decently, 
    # but strictly we should load dataset ONCE outside objective.
    # To do this, we'll use a global variable or class.
    
    # 3. Train Model
    trainer = ModelTrainer(seed=42)
    # We use global 'dataset'
    trainer.train(dataset, params=params)
    
    # 4. Evaluate (Validation Set)
    # We need to manually slice a validation set or use the test set defined in config?
    # Let's say we optimize for 2021-2022 performance (Validation).
    # And Train on 2015-2020.
    
    # Current dataset splitting in delorean code is roughly:
    # Train: Start -> End (usually split by Qlib DatasetH handler args)
    
    # We will use the 'test' segment of the dataset object as validation here.
    pred = trainer.predict(dataset)
    
    # Calculate IC (Information Coefficient)
    # Get labels
    # DatasetH prepare gives us (feature, label)
    df_test = dataset.prepare("test", col_set=["label"])
    
    # Align
    # pred index: (datetime, instrument)
    # label index: (datetime, instrument)
    
    combined = pd.concat([df_test['label'], pred], axis=1)
    combined.columns = ['label', 'score']
    combined = combined.dropna()
    
    # Rank IC
    ic = combined.groupby(level='datetime').apply(lambda x: x['label'].corr(x['score'], method='spearman'))
    mean_ic = ic.mean()
    
    logger.info(f"Trial {trial.number}: Mean IC = {mean_ic:.4f}")
    
    return mean_ic

if __name__ == "__main__":
    # Initialize Qlib
    qlib.init(provider_uri=QLIB_PROVIDER_URI, region="cn")
    
    # Load Data ONCE
    # We need to define specific Train/Valid periods for tuning
    # Let's define a Validation Validation Split
    VALID_START = "2022-01-01"
    VALID_END = "2022-12-31"
    TRAIN_END = "2021-12-31"
    
    logger.info("Loading Data...")
    loader = ETFDataLoader(ETF_LIST, start_time=START_TIME, end_time=VALID_END)
    # We construct dataset with Valid as 'test'
    dataset = loader.load_data(train_end=TRAIN_END, test_start=VALID_START)
    
    logger.info("Starting Optimization...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20) # Start with 20 trials for MVP
    
    logger.info("Best params:")
    logger.info(study.best_params)
    logger.info(f"Best IC: {study.best_value}")
    
    # Save best params
    import json
    with open("best_params.json", "w") as f:
        json.dump(study.best_params, f, indent=4)
