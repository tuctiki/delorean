
import os
import qlib
from typing import Optional, Dict, Any, Tuple
import pandas as pd

from delorean.conf import QLIB_PROVIDER_URI, QLIB_REGION, DEFAULT_EXPERIMENT_NAME
from delorean.data import ETFDataLoader
from delorean.model import ModelTrainer
from delorean.feature_selection import FeatureSelector
from delorean.signals import smooth_predictions
from delorean.utils import fix_seed
from qlib.workflow import R

class OptimizationConfig:
    """Configuration for Strategy Optimization (Stage 2)"""
    def __init__(self, use_alpha158: bool = False, use_hybrid: bool = False, 
                 smooth_window: int = 10, target_vol: float = 0.20):
        self.use_alpha158 = use_alpha158
        self.use_hybrid = use_hybrid
        self.smooth_window = smooth_window
        self.target_vol = target_vol

class StrategyRunner:
    """
    Unified Runner for Strategy Training and Prediction.
    Encapsulates Qlib initialization, data loading, model training, and signal generation.
    """
    def __init__(self, seed: int = 42, experiment_name: str = DEFAULT_EXPERIMENT_NAME):
        self.seed = seed
        self.experiment_name = experiment_name
        self.dataset = None
        self.pred = None
        self.model_trainer = None

    def initialize(self):
        """Initialize Qlib and Seed."""
        fix_seed(self.seed)
        qlib.init(provider_uri=QLIB_PROVIDER_URI, region=QLIB_REGION, kernels=1)

    def load_data(self, start_time: str, end_time: str, train_end_time: str, test_start_time: str,
                  label_horizon: int = 1, use_alpha158: bool = False, use_hybrid: bool = False):
        """Load Data using ETFDataLoader."""
        loader = ETFDataLoader(
            use_alpha158=use_alpha158,
            use_hybrid=use_hybrid,
            label_horizon=label_horizon,
            start_time=start_time,
            end_time=end_time
        )
        self.dataset = loader.load_data(
            train_start=start_time,
            train_end=train_end_time,
            test_start=test_start_time,
            test_end=end_time
        )
        return self.dataset

    def train_model(self, optimize_config: Optional[OptimizationConfig] = None):
        """
        Train the model (and optionally run Stage 2 optimization).
        Returns predictions.
        """
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load_data() first.")

        self.model_trainer = ModelTrainer(seed=self.seed)
        
        # Stage 1: Initial Training
        print("Stage 1: Initial Model Training...")
        self.model_trainer.train(self.dataset)
        self.pred = self.model_trainer.predict(self.dataset)
        
        # Stage 2: Optimization (Feature Selection)
        if optimize_config and (optimize_config.use_alpha158 or optimize_config.use_hybrid):
            print("\n[Optimization] Performing Feature Selection...")
            feature_imp = self.model_trainer.get_feature_importance(self.dataset)
            top_features_initial = feature_imp['feature'].head(40).tolist()
            
            # Filter
            top_features = FeatureSelector.filter_by_correlation(self.dataset, top_features_initial, threshold=0.95)
            top_features = top_features[:20]
            print(f"Final Selected Features ({len(top_features)}): {top_features}")
            
            print("Stage 2: Retraining with Selected Features...")
            self.model_trainer.train(self.dataset, selected_features=top_features)
            self.pred = self.model_trainer.predict(self.dataset)

        # Smoothing
        if optimize_config and optimize_config.smooth_window > 0:
            print(f"Applying {optimize_config.smooth_window}-day EWMA Signal Smoothing...")
            self.pred = smooth_predictions(self.pred, halflife=optimize_config.smooth_window)
            
        return self.pred

    from contextlib import contextmanager

    @contextmanager
    def run_experiment(self, params: Dict[str, Any]):
        """
        Wrap execution in an MLflow run.
        """
        import mlflow
        if mlflow.active_run():
            mlflow.end_run()
            
        with R.start(experiment_name=self.experiment_name) as recorder:
            # Ensure params are strings
            log_params = {k: str(v) for k, v in params.items()}
            R.log_params(**log_params)
            
            if self.pred is None:
                # Assuming train_model or similar logic is called inside or before
                # For flexible usage, we might just return the recorder context
                pass
            yield recorder
