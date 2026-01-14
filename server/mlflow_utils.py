"""
MLflow utilities for reading experiment data from the mlruns directory.
Provides shared helper functions to eliminate duplicate path-finding logic.
"""

import os
import math
from typing import Optional, Dict, Any, List

from delorean.config import DEFAULT_EXPERIMENT_NAME


def get_project_root() -> str:
    """Get the project root directory."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_mlruns_path() -> str:
    """Get the path to the mlruns directory."""
    return os.path.join(get_project_root(), "mlruns")


def get_default_experiment_path() -> Optional[str]:
    """
    Find and return the path to the default experiment folder.
    
    Returns:
        The absolute path to the default experiment folder, or None if not found.
    """
    mlruns_path = get_mlruns_path()
    
    if not os.path.exists(mlruns_path):
        return None
    
    for d in os.listdir(mlruns_path):
        full_path = os.path.join(mlruns_path, d)
        if os.path.isdir(full_path) and d.isdigit():
            meta_path = os.path.join(full_path, "meta.yaml")
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, "r") as f:
                        for line in f:
                            if line.strip().startswith("name:"):
                                exp_name = line.split(":", 1)[1].strip()
                                if exp_name == DEFAULT_EXPERIMENT_NAME:
                                    return full_path
                except Exception:
                    pass
    return None


def read_run_params(run_path: str) -> Dict[str, str]:
    """
    Read parameters from an MLflow run directory.
    
    Args:
        run_path: Path to the run directory.
        
    Returns:
        Dictionary of parameter names to values.
    """
    params = {}
    params_path = os.path.join(run_path, "params")
    
    if os.path.exists(params_path):
        for pf in os.listdir(params_path):
            try:
                with open(os.path.join(params_path, pf), "r") as f:
                    params[pf] = f.read().strip()
            except Exception:
                pass
    
    return params


def read_run_metrics(run_path: str) -> Dict[str, Optional[float]]:
    """
    Read metrics from an MLflow run directory.
    
    Args:
        run_path: Path to the run directory.
        
    Returns:
        Dictionary of metric names to values (None for invalid values).
    """
    metrics = {}
    metrics_path = os.path.join(run_path, "metrics")
    
    if os.path.exists(metrics_path):
        for mf in os.listdir(metrics_path):
            try:
                with open(os.path.join(metrics_path, mf), "r") as f:
                    parts = f.read().strip().split()
                    if len(parts) >= 2:
                        val = float(parts[1])
                        if math.isnan(val) or math.isinf(val):
                            val = None
                        metrics[mf] = val
            except Exception:
                pass
    
    return metrics


def list_runs(experiment_path: str) -> List[str]:
    """
    List all run IDs in an experiment directory.
    
    Args:
        experiment_path: Path to the experiment directory.
        
    Returns:
        List of run IDs (directory names that look like MLflow run IDs).
    """
    runs = []
    
    if os.path.exists(experiment_path):
        for run_id in os.listdir(experiment_path):
            run_path = os.path.join(experiment_path, run_id)
            # Run IDs are UUIDs (32 chars without hyphens, or 36 with)
            if os.path.isdir(run_path) and len(run_id) >= 20:
                runs.append(run_id)
    
    return runs
