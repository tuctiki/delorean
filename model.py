from qlib.contrib.model.gbdt import LGBModel
from qlib.workflow import R
import pandas as pd
import matplotlib.pyplot as plt

class ModelTrainer:
    def __init__(self):
        self.model = None

    def train(self, dataset):
        print("\nUsing LightGBM Model (Optimized - Exp 8)...")
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

    def predict(self, dataset):
        print("Generating test set predictions...")
        pred = self.model.predict(dataset)
        R.save_objects(pred=pred) # Save to current recorder if active
        print("\nTest prediction sample (head 20):\n", pred.head(20))
        return pred

    def get_feature_importance(self, dataset):
        feature_importance = pd.DataFrame({
            'feature': dataset.prepare("train", col_set="feature").columns,
            'importance': self.model.get_feature_importance()
        }).sort_values('importance', ascending=False)
        return feature_importance
