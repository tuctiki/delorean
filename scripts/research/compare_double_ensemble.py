
import qlib
import pandas as pd
import numpy as np
from qlib.contrib.model.gbdt import LGBModel
from qlib.contrib.model.double_ensemble import DEnsembleModel
from qlib.contrib.evaluate import risk_analysis
from qlib.data.dataset import DatasetH
import matplotlib.pyplot as plt
import os
import sys

# Add workspace to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from delorean.conf import QLIB_PROVIDER_URI, QLIB_REGION, ETF_LIST
from delorean.data import ETFDataLoader
from delorean.utils import run_standard_backtest

def compare_models():
    qlib.init(provider_uri=QLIB_PROVIDER_URI, region=QLIB_REGION)
    
    # 1. Load Data (9-factor library)
    print("\n[1] Loading Data...")
    from delorean.conf import START_TIME, END_TIME, TRAIN_END_TIME, TEST_START_TIME, ETF_LIST
    from delorean.data.handlers import ETFDataHandler
    
    # Define custom segments for robust testing:
    # Train: 2015-2021 (7 years)
    # Valid: 2022 (1 year)
    # Test:  2023-2025 (3 years)
    segments = {
        "train": ("2015-01-01", "2021-12-31"),
        "valid": ("2022-01-01", "2022-12-31"),
        "test":  ("2023-01-01", "2025-12-31")
    }
    
    handler = ETFDataHandler(
        instruments=ETF_LIST,
        start_time="2015-01-01",
        end_time="2025-12-31",
        label_horizon=1
    )
    
    dataset = DatasetH(handler=handler, segments=segments)
    print(f"Dataset Segments: {segments}")
    
    # 2. Setup Models
    models = {
        "LightGBM (Baseline)": LGBModel(
            loss="mse",
            learning_rate=0.02,
            max_depth=5,
            num_leaves=19,
            lambda_l1=0.1,
            lambda_l2=0.1,
            seed=42
        ),
        "DoubleEnsemble": DEnsembleModel(
            base_model="gbm", # Use standard gbm
            loss="mse",
            num_models=3,    # Reduced for stability/speed during testing
            bins_sr=5,       # Sane default
            bins_fs=3,       # Reduced from 5, since we only have 9 features
            sample_ratios=[0.8, 0.6, 0.4],
            decay=1.0,       # Fixed: cannot be None
            enable_sr=True,
            enable_fs=True,
            # Base model params
            learning_rate=0.03,
            max_depth=4,
            num_leaves=15,
            seed=42,
            early_stopping_rounds=50,
            epochs=100
        )
    }
    
    results = {}
    
    # 3. Train and Predict
    for name, model in models.items():
        print(f"\n--- Running {name} ---")
        model.fit(dataset)
        pred = model.predict(dataset)
        
        # Calculate Rank IC
        df_test = dataset.prepare("test", col_set=["label"])
        y_test = df_test["label"].iloc[:, 0]
        eval_df = pd.DataFrame({"score": pred, "label": y_test}).dropna()
        
        daily_ic = eval_df.groupby(level='datetime').apply(
            lambda x: x["score"].corr(x["label"], method="spearman")
        )
        mean_ic = daily_ic.mean()
        ic_ir = mean_ic / daily_ic.std() if daily_ic.std() != 0 else 0
        
        # Run Backtest
        print(f"  > Running Backtest for {name}...")
        report, _ = run_standard_backtest(
            pred=pred,
            topk=4,
            use_regime_filter=True
        )
        
        risks = risk_analysis(report["return"], freq="day")
        sharpe = risks.loc["information_ratio", "risk"]
        ann_ret = risks.loc["annualized_return", "risk"]
        max_dd = risks.loc["max_drawdown", "risk"]
        
        results[name] = {
            "Rank IC": mean_ic,
            "ICIR": ic_ir,
            "Sharpe": sharpe,
            "Ann. Return": ann_ret,
            "Max Drawdown": max_dd,
            "pred": pred
        }
        
    # 4. Print Summary
    print("\n" + "="*50)
    print(f"{'Model Comparison Summary (2023-2025)':^50}")
    print("="*50)
    summary_df = pd.DataFrame(results).T[["Rank IC", "ICIR", "Sharpe", "Ann. Return", "Max Drawdown"]]
    print(summary_df)
    
    # 5. Plot Comparison
    plt.figure(figsize=(12, 6))
    for name, res in results.items():
        cum_ret = (1 + res["pred"].groupby(level='datetime').mean()).cumprod() # Just a directional check
        # Better: plot the actual strategy return
        # report_df = run_standard_backtest result
        # But we need cumulative return of strategy
    
    print("\nDone.")

if __name__ == "__main__":
    compare_models()
