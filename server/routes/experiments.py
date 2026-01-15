"""
Experiment-related API routes.
Handles MLflow experiment listing, run details, and artifacts.
"""

import os
import pandas as pd
from fastapi import APIRouter, HTTPException

from server.mlflow_utils import (
    get_project_root,
    get_default_experiment_path,
    read_run_params,
    read_run_metrics,
    list_runs,
)

router = APIRouter(prefix="/api", tags=["experiments"])


@router.get("/experiments")
def list_experiments_route():
    """
    List all runs from the default experiment (ETF_Strategy).
    Returns a flat list of runs with their params and metrics.
    """
    try:
        default_exp_path = get_default_experiment_path()
        
        if not default_exp_path:
            return []  # Default experiment not found
        
        runs = []
        for run_id in list_runs(default_exp_path):
            run_path = os.path.join(default_exp_path, run_id)
            
            # Get creation time
            creation_time = os.path.getmtime(run_path)
            creation_str = pd.Timestamp(creation_time, unit='s').strftime('%Y-%m-%d %H:%M:%S')
            
            runs.append({
                "id": run_id,
                "name": f"Run {creation_str}",
                "artifact_location": run_path,
                "creation_time": creation_str,
                "timestamp": creation_time,
                "params": read_run_params(run_path),
                "metrics": read_run_metrics(run_path)
            })
        
        # Sort by Timestamp Descending (newest first)
        runs.sort(key=lambda x: x["timestamp"], reverse=True)
        return runs
    except Exception as e:
        import traceback
        return [{"id": "error", "name": f"Error: {str(e)}", "timestamp": 0, "metrics": {}, "creation_time": traceback.format_exc()}]


@router.get("/experiment_results")
def get_experiment_results_route():
    """
    Get latest backtest results, normalizing field names for frontend.
    """
    import json
    
    root = get_project_root()
    path = os.path.join(root, "artifacts", "experiment_results.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            data = json.load(f)
        
        # Normalize field names to match frontend expectations
        normalized = {
            "sharpe": data.get("sharpe_ratio"),
            "annualized_return": data.get("annual_return"),
            "max_drawdown": data.get("max_drawdown"),
            "win_rate": data.get("win_rate"),
            "daily_turnover": data.get("daily_turnover"),
            "annualized_turnover": data.get("annualized_turnover"),
            "trading_days": data.get("trading_days"),
            "total_days": data.get("total_days"),
            "trading_frequency": data.get("trading_frequency"),
            "description": "Latest Backtest",
            "period": data.get("timestamp", "N/A"),
        }
        
        # Load chart data from backtest report if available
        report_path = os.path.join(root, "artifacts", "backtest_report.pkl")
        if os.path.exists(report_path):
            try:
                report = pd.read_pickle(report_path)
                # Create chart data with cumulative returns
                cum_return = (1 + report["return"]).cumprod()
                bench_cum = (1 + report["bench"]).cumprod()
                
                chart_data = []
                for idx in cum_return.index:
                    date_str = idx.strftime("%Y-%m-%d") if hasattr(idx, 'strftime') else str(idx)
                    chart_data.append({
                        "date": date_str,
                        "strategy": round(float(cum_return.loc[idx]), 4),
                        "benchmark": round(float(bench_cum.loc[idx]), 4)
                    })
                # Sample to reduce data points for frontend performance
                if len(chart_data) > 200:
                    step = len(chart_data) // 200
                    chart_data = chart_data[::step]
                normalized["chart_data"] = chart_data
            except Exception as e:
                print(f"Error loading chart data: {e}")
        
        return normalized
    return {}


@router.get("/experiments/{run_id}")
def get_experiment_details_route(run_id: str):
    """
    Get details for a specific run by its ID.
    """
    default_exp_path = get_default_experiment_path()
    
    if not default_exp_path:
        raise HTTPException(status_code=404, detail="Default experiment not found")
    
    run_path = os.path.join(default_exp_path, run_id)
    if not os.path.exists(run_path):
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
        
    return {
        "id": run_id,
        "artifact_location": run_path,
        "params": read_run_params(run_path),
        "metrics": read_run_metrics(run_path),
        "status": "FINISHED"
    }


@router.get("/experiments/{run_id}/image")
def get_experiment_image_route(run_id: str, name: str):
    """
    Get an image artifact from a specific run.
    """
    from fastapi.responses import FileResponse
    
    default_exp_path = get_default_experiment_path()
    
    if not default_exp_path:
        raise HTTPException(status_code=404, detail="Default experiment not found")
    
    run_path = os.path.join(default_exp_path, run_id)
    if not os.path.exists(run_path):
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    
    # Check standard artifacts location
    possible_paths = [
        os.path.join(run_path, "artifacts", name),
        os.path.join(run_path, name),
    ]
    
    for p in possible_paths:
        if os.path.exists(p):
            return FileResponse(p)
            
    raise HTTPException(status_code=404, detail="Image not found")
