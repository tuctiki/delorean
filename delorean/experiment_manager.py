from typing import Dict, Any, Optional
import pandas as pd
from qlib.workflow import R
import yaml
import os

class ExperimentManager:
    """
    Manages experiment logging and artifact saving using Qlib's Recorder.
    """
    def __init__(self):
        pass

    def log_config(self, args: Any, model_params: Dict[str, Any], strategy_params: Dict[str, Any], other_params: Optional[Dict[str, Any]] = None) -> None:
        """
        Log experiment configuration to the Recorder.

        Args:
            args: Command line arguments namespace.
            model_params: Dictionary of model hyperparameters.
            strategy_params: Dictionary of strategy parameters (e.g., topk).
            other_params: Any other relevant parameters.
        """
        # 1. Log Flattened Parameters for UI tracking (MLflow/Qlib)
        # Prefix keys to avoid collisions
        flat_params = {}
        
        # Args
        if args:
            for k, v in vars(args).items():
                flat_params[f"arg_{k}"] = v
                
        # Model Params
        for k, v in model_params.items():
            flat_params[f"model_{k}"] = v
            
        # Strategy Params
        for k, v in strategy_params.items():
            flat_params[f"strat_{k}"] = v
            
        if other_params:
            for k, v in other_params.items():
                flat_params[f"other_{k}"] = v

        # Log to Qlib Recorder (logs to MLflow/FileBackend)
        R.log_params(**flat_params)
        
        # 2. Save Full Config as YAML Artifact for easy review
        full_config = {
            "arguments": vars(args) if args else {},
            "model_parameters": model_params,
            "strategy_parameters": strategy_params,
            "other_parameters": other_params or {}
        }
        
        # Save as a dict object (which Qlib pickles)
        R.save_objects(config=full_config)
        
        # Also save as readable YAML
        recorder_root = R.get_recorder().get_local_dir()
        if recorder_root:
            yaml_path = os.path.join(recorder_root, "experiment_config.yaml")
            with open(yaml_path, 'w') as f:
                yaml.dump(full_config, f, default_flow_style=False)
            print(f"Experiment config saved to: {yaml_path}")

    def save_report(self, report: pd.DataFrame, positions: Dict[Any, Any]) -> None:
        """
        Save backtest report and positions.
        """
        R.save_objects(report=report, positions=positions)
        print("Backtest report and positions saved to Recorder artifacts.")
