from typing import Any, Optional
from qlib.contrib.model.gbdt import LGBModel
from qlib.data.dataset import DatasetH
from qlib.workflow import R
import pandas as pd
import matplotlib.pyplot as plt

class ModelTrainer:
    """
    Encapsulates the training and prediction logic for the LightGBM model.
    """
    def __init__(self):
        """
        Initialize the ModelTrainer.
        
        The model is initialized as None and will be instantiated during training
        with optimized hyperparameters.
        """
        self.model: Optional[LGBModel] = None

    def train(self, dataset: DatasetH) -> None:
        """
        Train the LightGBM model using the provided Qlib dataset.

        Args:
            dataset (DatasetH): The Qlib dataset object containing train and test segments.
        """
        print("\nUsing LightGBM Model (Optimized - Exp 8)...")
        # Hyperparameters tuned in Experiment 8 for stability and reduced turnover
        self.model = LGBModel(
            loss="mse",
            colsample_bytree=0.887,
            learning_rate=0.05,          # Accelerated convergence
            subsample=0.7,
            lambda_l1=0.5,               # Relaxed regularization
            lambda_l2=0.5,
            max_depth=-1,
            num_leaves=31,               # Increased complexity
            min_data_in_leaf=30,         # Reduced minimum data
            early_stopping_rounds=100,   # Increased patience
            num_boost_round=1000,        # More iterations
        )
        
        print("Starts training...")
        # Note: Recorder context should be handled effectively by the caller if needed, 
        # but LGBModel handles fit(dataset) natively with DatasetH.
        self.model.fit(dataset)

    def predict(self, dataset: DatasetH) -> pd.Series:
        """
        Generate predictions for the test segment of the dataset.

        Args:
            dataset (DatasetH): The Qlib dataset object.

        Returns:
            pd.Series: A Series of prediction scores indexed by (datetime, instrument).
        """
        print("Generating test set predictions...")
        pred = self.model.predict(dataset)
        R.save_objects(pred=pred) # Save to current recorder if active
        print("\nTest prediction sample (head 20):\n", pred.head(20))
        return pred

    def get_feature_importance(self, dataset: DatasetH) -> pd.DataFrame:
        """
        Extract and return feature importance from the trained model.

        Args:
            dataset (DatasetH): The Qlib dataset object (used to retrieve column names).

        Returns:
            pd.DataFrame: A DataFrame with 'feature' and 'importance' columns, sorted by importance.
        """
        # Ensure the model is trained
        if self.model is None:
            raise ValueError("Model has not been trained yet.")

        feature_importance = pd.DataFrame({
            'feature': dataset.prepare("train", col_set="feature").columns,
            'importance': self.model.get_feature_importance()
        }).sort_values('importance', ascending=False)
        return feature_importance
