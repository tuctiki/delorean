from typing import Any, Optional, List, Dict
from qlib.contrib.model.gbdt import LGBModel
from qlib.data.dataset import DatasetH
from qlib.workflow import R
import pandas as pd
import lightgbm as lgb
import numpy as np

class ModelTrainer:
    """
    Encapsulates the training and prediction logic for the LightGBM model.
    """
    def __init__(self):
        """
        Initialize the ModelTrainer.
        """
        self.model: Any = None # Can be LGBModel or lgb.Booster
        self.selected_features: Optional[List[str]] = None
        self.params: Dict[str, Any] = {}

    def train(self, dataset: DatasetH, selected_features: Optional[List[str]] = None) -> None:
        """
        Train the LightGBM model.
        
        Args:
            dataset (DatasetH): The Qlib dataset object.
            selected_features (List[str], optional): If provided, train ONLY on these features using native LightGBM.
        """
        if selected_features:
            print(f"\nTraining Optimized LightGBM on {len(selected_features)} selected features...")
            self.selected_features = selected_features
            
            # fast data extraction
            df_train = dataset.prepare("train", col_set=["feature", "label"])
            X_train = df_train["feature"][selected_features]
            y_train = df_train["label"]
            
            # Optimized params for Stage 2 (smaller refined model)
            params = {
                "objective": "regression",
                "metric": "mse",
                "learning_rate": 0.03, # Slower learning for robustness
                "num_leaves": 15,      # Smaller trees
                "colsample_bytree": 0.6,
                "subsample": 0.6,
                "reg_alpha": 1.0,      # Stronger L1
                "reg_lambda": 1.0,     # Stronger L2
                "n_jobs": -1,
                "verbosity": -1
            }
            self.params = params
            
            # Create native dataset
            dtrain = lgb.Dataset(X_train, label=y_train)
            
            # Train
            self.model = lgb.train(
                params,
                dtrain,
                num_boost_round=1000,
                callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)],
                valid_sets=[dtrain] # Use train as valid for simplicity/convergence check or split
            )
            
        else:
            print("\nUsing LightGBM Model (Optimized - Exp 8)...")
            self.selected_features = None
            # Hyperparameters tuned in Experiment 8
            self.params = {
                "loss": "mse",
                "colsample_bytree": 0.887,
                "learning_rate": 0.05,
                "subsample": 0.7,
                "lambda_l1": 0.5,
                "lambda_l2": 0.5,
                "max_depth": -1,
                "num_leaves": 31,
                "min_data_in_leaf": 30,
                "early_stopping_rounds": 100,
                "num_boost_round": 1000,
            }
            self.model = LGBModel(**self.params)
            print("Starts training...")
            self.model.fit(dataset)

    def predict(self, dataset: DatasetH) -> pd.Series:
        """
        Generate predictions for the test segment.
        """
        print("Generating test set predictions...")
        
        if self.selected_features:
            # Native Booster Prediction
            df_test = dataset.prepare("test", col_set="feature")
            X_test = df_test[self.selected_features]
            pred_values = self.model.predict(X_test)
            # Align index
            pred = pd.Series(pred_values, index=X_test.index)
        else:
            # Qlib LGBModel Prediction
            pred = self.model.predict(dataset)
            
        R.save_objects(pred=pred) 
        print("\nTest prediction sample (head 20):\n", pred.head(20))
        return pred

    def get_feature_importance(self, dataset: DatasetH) -> pd.DataFrame:
        """
        Extract feature importance.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")

        if self.selected_features:
            # Native Booster
            importance = self.model.feature_importance(importance_type='gain')
            feature_names = self.selected_features
        else:
            # Qlib LGBModel
            importance = self.model.get_feature_importance()
            feature_names = dataset.prepare("train", col_set="feature").columns
            
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        return feature_importance

    def get_params(self) -> Dict[str, Any]:
        """Return the parameters used for training."""
        return self.params
