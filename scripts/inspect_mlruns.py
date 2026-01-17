
import mlflow
import pandas as pd
from typing import List, Dict

def inspect_recent_runs(experiment_name: str = "delorean_experiment", limit: int = 10):
    """
    List recent runs and their metrics.
    """
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            print(f"Experiment '{experiment_name}' not found.")
            return

        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            max_results=limit,
            order_by=["start_time DESC"]
        )
        
        print(f"--- Recent {limit} Runs for '{experiment_name}' ---")
        for _, run in runs.iterrows():
            run_id = run["run_id"]
            start_time = run["start_time"]
            status = run["status"]
            
            # metrics columns usually start with 'metrics.'
            metrics = {k.replace("metrics.", ""): v for k, v in run.items() if k.startswith("metrics.") and pd.notnull(v)}
            # params columns usually start with 'params.'
            params = {k.replace("params.", ""): v for k, v in run.items() if k.startswith("params.") and pd.notnull(v)}
            
            print(f"Run ID: {run_id}")
            print(f"  Date: {start_time}")
            print(f"  Status: {status}")
            print(f"  Metrics: {metrics}")
            # print(f"  Params: {params}") # optional, too verbose maybe
            print("-" * 40)
            
            if "rank_ic" in metrics and metrics["rank_ic"] < 0:
                 print("  >>> DETECTED NEGATIVE RANK IC <<<")

    except Exception as e:
        print(f"Error inspecting runs: {e}")

if __name__ == "__main__":
    # You might need to adjust the tracking URI if it's set in delorean/conf.py or environment
    # Assuming default or local mlruns
    # Check if there is a specific tracking URI in conf
    import sys
    import os
    sys.path.append(os.getcwd())
    try:
        from delorean.conf import DEFAULT_EXPERIMENT_NAME
    except ImportError:
        DEFAULT_EXPERIMENT_NAME = "delorean_experiment"
        
    inspect_recent_runs(DEFAULT_EXPERIMENT_NAME)
