
import qlib
import pandas as pd
import numpy as np
import os
import sys
from qlib.contrib.model.double_ensemble import DEnsembleModel
from qlib.contrib.evaluate import risk_analysis
from qlib.data.dataset import DatasetH
import matplotlib.pyplot as plt

# Add workspace to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from delorean.conf import QLIB_PROVIDER_URI, QLIB_REGION, ETF_LIST
from delorean.data.handlers import ETFDataHandler
from delorean.utils import run_standard_backtest
import delorean.alphas.factors as factors_module

# Define the two factor libraries
FULL_9_FACTORS = [
    ("$close / Ref($close, 60) - 1", "MOM60"),
    ("$close / Ref($close, 120) - 1", "MOM120"),
    ("($close - $open) / (Abs($open - Ref($close, 1)) + 0.001)", "Gap_Fill"),
    ("Sum(If($close > Ref($close, 1), 1, 0), 10) / 10", "Mom_Persistence"),
    ("($close / Ref($close, 5) - 1) - (Ref($close, 5) / Ref($close, 10) - 1)", "Acceleration"),
    ("Mean(Corr($close / Ref($close, 1), $volume / Ref($volume, 1), 10), 5)", "Vol_Price_Div"),
    ("Mean(Sum((($close - $low) - ($high - $close)) / ($high - $low + 0.001) * $volume, 20) / Sum($volume, 20), 5)", "Money_Flow_20"),
    ("Corr($close / Ref($close, 1) - 1, (Mean(If($close > Ref($close, 1), $close - Ref($close, 1), 0), 14) / (Mean(Abs($close - Ref($close, 1)), 14) + 0.0001)), 10)", "RSI_Divergence"),
    ("-1 * (Sum(-1 * (Log($open)), 5) + Std($high, 5))", "Alpha_Gen_8"),
]

# REFINED 5: Pruned and sign-flipped Vol_Price_Div
REFINED_5_FACTORS = [
    ("$close / Ref($close, 60) - 1", "MOM60"),
    ("$close / Ref($close, 120) - 1", "MOM120"),
    # Vol_Price_Div: SIGN FLIPPED to negative correlation (mean-reverting trait in choppy markets)
    ("Mean(-1 * Corr($close / Ref($close, 1), $volume / Ref($volume, 1), 10), 5)", "Vol_Price_Div_Rev"),
    ("Mean(Sum((($close - $low) - ($high - $close)) / ($high - $low + 0.001) * $volume, 20) / Sum($volume, 20), 5)", "Money_Flow_20"),
    ("-1 * (Sum(-1 * (Log($open)), 5) + Std($high, 5))", "Alpha_Gen_8"),
]

def run_experiment(factor_list, name):
    print(f"\n>>> Running Experiment: {name}")
    
    # Monkeypatch the factor library
    factors_module.PRODUCTION_FACTORS = factor_list
    
    # Initialize Data
    handler = ETFDataHandler(
        instruments=ETF_LIST,
        start_time="2015-01-01",
        end_time="2025-12-31",
        label_horizon=1
    )
    
    segments = {
        "train": ("2015-01-01", "2021-12-31"),
        "valid": ("2022-01-01", "2022-12-31"),
        "test":  ("2023-01-01", "2025-12-31")
    }
    
    dataset = DatasetH(handler=handler, segments=segments)
    
    # Setup DoubleEnsemble
    model = DEnsembleModel(
        base_model="gbm",
        loss="mse",
        num_models=3,
        bins_sr=5,
        bins_fs=min(3, len(factor_list) // 2 + 1), # Adjust FS bins for small libraries
        sample_ratios=[0.8, 0.6, 0.4] if len(factor_list) > 4 else [0.8, 0.7],
        decay=1.0,
        enable_sr=True,
        enable_fs=True if len(factor_list) > 3 else False,
        learning_rate=0.03,
        max_depth=4,
        num_leaves=15,
        seed=42,
        early_stopping_rounds=50,
        epochs=100
    )
    
    model.fit(dataset)
    pred = model.predict(dataset)
    
    # Calculate Metrics
    df_test = dataset.prepare("test", col_set=["label"])
    y_test = df_test["label"].iloc[:, 0]
    eval_df = pd.DataFrame({"score": pred, "label": y_test}).dropna()
    daily_ic = eval_df.groupby(level='datetime').apply(lambda x: x["score"].corr(x["label"], method="spearman"))
    
    # Backtest
    report, _ = run_standard_backtest(pred=pred, topk=4, use_regime_filter=True)
    risks = risk_analysis(report["return"], freq="day")
    
    return {
        "Rank IC": daily_ic.mean(),
        "ICIR": daily_ic.mean() / daily_ic.std() if daily_ic.std() != 0 else 0,
        "Sharpe": risks.loc["information_ratio", "risk"],
        "Ann. Return": risks.loc["annualized_return", "risk"],
        "Max Drawdown": risks.loc["max_drawdown", "risk"]
    }

def main():
    qlib.init(provider_uri=QLIB_PROVIDER_URI, region=QLIB_REGION)
    
    results = {}
    
    # 1. Base Case (Full 9)
    results["Full 9 Factors"] = run_experiment(FULL_9_FACTORS, "Full 9 Factors")
    
    # 2. Refined Case (Refined 5)
    results["Refined 5 Factors"] = run_experiment(REFINED_5_FACTORS, "Refined 5 Factors")
    
    # Summary Table
    summary_df = pd.DataFrame(results).T
    print("\n" + "="*60)
    print(f"{'FACTOR REFINEMENT COMPARISON (2023-2025)':^60}")
    print("="*60)
    print(summary_df.round(4))
    
if __name__ == "__main__":
    main()
