from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
from qlib.data.dataset.processor import CSZScoreNorm, DropnaLabel
from qlib.contrib.model.gbdt import LGBModel
from qlib.workflow import R
import pandas as pd
import matplotlib.pyplot as plt
from constants import START_TIME, END_TIME, TRAIN_END_TIME, TEST_START_TIME, ETF_LIST

class ETFDataHandler(DataHandlerLP):
    def __init__(self, instruments, start_time, end_time, **kwargs):
        data_loader_config = {
            "feature": [
                "Log(Mean($volume * $close, 20))",                  # Market Cap / Liquidity Proxy
                "$close / Ref($close, 20) - 1",                 # 20-day Momentum
                "$close / Ref($close, 60) - 1",                 # 60-day Momentum
                "$close / Ref($close, 120) - 1",               # 120-day Momentum
                "($close / Ref($close, 5) - 1) * -1",                # 5-day Reversal
                "Std($close / Ref($close, 1) - 1, 20)",         # 20-day Volatility
                "Std($close / Ref($close, 1) - 1, 60)",         # 60-day Volatility
                "$volume / Mean($volume, 20)",                       # Short-term Volume Ratio
                "$volume / Mean($volume, 60)",                       # Medium-term Volume Ratio
                "Skew($close / Ref($close, 1) - 1, 20)",       # 20-day Skewness
            ],
            "label": [
                "Ref($close, -1) / $close - 1"  # Next Day Return
            ],
        }
        data_loader = {
            "class": "QlibDataLoader",
            "kwargs": {
                "config": data_loader_config,
                "freq": "day",
            }
        }
        processors = [
            DropnaLabel(),
            CSZScoreNorm(fields_group="feature"),
        ]
        super().__init__(
            data_loader=data_loader,
            learn_processors=processors,
            **kwargs
        )

class ETFStrategy:
    def __init__(self):
        self.handler = None
        self.dataset = None
        self.model = None

    def load_data(self):
        print("Initializing ETF DataHandler...")
        self.handler = ETFDataHandler(
            instruments=ETF_LIST,
            start_time=START_TIME,
            end_time=END_TIME,
        )
        
        segments = {
            "train": (START_TIME, TRAIN_END_TIME),
            "test": (TEST_START_TIME, END_TIME),
        }
        
        print("Creating Dataset...")
        self.dataset = DatasetH(
            handler=self.handler,
            segments=segments,
        )
        self._print_data_stats()

    def _print_data_stats(self):
        train_features = self.dataset.prepare("train", col_set="feature")
        print(f"Training set shape: {train_features.shape}")
        print(f"Features: {list(train_features.columns)}")
        print("\nTraining set correlation:\n", train_features.corr().round(2))

    def train_model(self):
        print("\nUsing LightGBM Model...")
        self.model = LGBModel(
            loss="mse",
            colsample_bytree=0.887,
            learning_rate=0.05,
            subsample=0.7,
            lambda_l1=1,
            lambda_l2=1,
            max_depth=-1,
            num_leaves=31,
            min_data_in_leaf=20,
            early_stopping_rounds=50,
        )
        
        print("Starts training...")
        with R.start(experiment_name="ETF_Strategy") as recorder:
            self.model.fit(self.dataset)
            self.predict(recorder)
            self.analyze_feature_importance()

    def predict(self, recorder):
        print("Generating test set predictions...")
        pred = self.model.predict(self.dataset)
        R.save_objects(pred=pred)
        print("\nTest prediction sample (head 20):\n", pred.head(20))

    def analyze_feature_importance(self):
        feature_importance = pd.DataFrame({
            'feature': self.dataset.prepare("train", col_set="feature").columns,
            'importance': self.model.get_feature_importance()
        }).sort_values('importance', ascending=False)

        print("\nTop 10 Feature Importance:\n", feature_importance.head(10))
        
        try:
            plt.figure(figsize=(10, 6))
            plt.bar(feature_importance['feature'][:10], feature_importance['importance'][:10])
            plt.xticks(rotation=45, ha='right')
            plt.title("LightGBM Factor Importance Top 10")
            plt.tight_layout()
            print("Feature importance plot generated (display skipped).")
        except Exception as e:
            print(f"Plotting failed: {e}")

    def run(self):
        self.load_data()
        self.train_model()
