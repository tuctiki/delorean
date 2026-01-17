
import pandas as pd
import qlib
from qlib.data import D
from qlib.contrib.model.gbdt import LGBModel
from qlib.contrib.data.handler import Alpha158
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from delorean.conf import QLIB_PROVIDER_URI, QLIB_REGION, ETF_LIST
from delorean.alphas.factors import get_production_factors
from delorean.data import ETFDataLoader

def debug_model():
    qlib.init(provider_uri=QLIB_PROVIDER_URI, region=QLIB_REGION)
    
    # 1. Load Data (Train on 2024-2025 to simulate what the live model sees)
    train_start = "2024-01-01"
    train_end = "2025-12-31"
    
    loader = ETFDataLoader(
        start_time=train_start,
        end_time=train_end,
        label_horizon=1
    )
    dataset = loader.load_data(
        train_start=train_start,
        train_end=train_end,
        test_start="2026-01-01", # Dummy, we just want to train
        test_end="2026-01-17"
    )
    
from delorean.conf.model import MODEL_PARAMS_STAGE1

def debug_model():
    qlib.init(provider_uri=QLIB_PROVIDER_URI, region=QLIB_REGION)
    
    # 1. Load Data
    train_start = "2024-01-01"
    train_end = "2025-12-31"
    
    loader = ETFDataLoader(
        start_time=train_start,
        end_time=train_end,
        label_horizon=1
    )
    dataset = loader.load_data(
        train_start=train_start,
        train_end=train_end,
        test_start="2026-01-01",
        test_end="2026-01-17"
    )
    
    # 2. Train Model with Production Params
    print("Training Model on 2024-2025 with Production Params...")
    params = MODEL_PARAMS_STAGE1.copy()
    params["seed"] = 42
    model = LGBModel(**params)
    model.fit(dataset)
    
    # 3. Inspect Feature Direction
    pred_train = model.predict(dataset, segment="train")
    df_train = dataset.prepare("train", col_set=["feature", "label"])
    
    if isinstance(pred_train, pd.Series):
        pred_train = pred_train.to_frame("score")
    
    common = df_train.index.intersection(pred_train.index)
    df_train = df_train.loc[common]
    pred_train = pred_train.loc[common]
    
    print("\n--- Effective Feature Direction (Train Set) ---")
    # Get expressions from registry
    expressions, names = get_production_factors()
    
    # Check if MultiIndex
    is_multi = isinstance(df_train.columns, pd.MultiIndex)
    
    for expr, name in zip(expressions, names):
        # Access feature column
        try:
            if is_multi:
                feat_col = df_train["feature"][expr]
                # Label might be ("label", "label") or just "label"?
                # Usually standard Qlib label is ("label", "label") if multiple labels, or just "label"?
                # Let's inspect columns or try both
                if "label" in df_train.columns:
                    label_col = df_train["label"]
                    # If label has sub-columns, take first?
                    if isinstance(label_col, pd.DataFrame):
                         label_col = label_col.iloc[:, 0]
                else:
                    # Try finding label in level 1?
                    label_col = df_train.xs("label", axis=1, level=0, drop_level=False)
            else:
                 feat_col = df_train[expr]
                 label_col = df_train["label"]
                 
            corr_pred = feat_col.corr(pred_train["score"])
            corr_label = feat_col.corr(label_col)
            
            print(f"{name:<20} | Pred Corr: {corr_pred: >7.4f} | Label Corr: {corr_label: >7.4f}")
            
            if corr_pred < 0 and corr_label > 0:
                 print(f"   >>> WARNING: Model is betting AGAINST {name} (Positive Signal, Negative Weight)")
                 
        except KeyError:
            print(f"Skipping {name} (Not found in dataset)")
        except Exception as e:
            print(f"Error checking {name}: {e}")

if __name__ == "__main__":
    debug_model()
