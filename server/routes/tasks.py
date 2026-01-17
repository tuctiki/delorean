"""
Background task API routes.
Handles daily task and backtest task management.
"""

import os
import sys
import subprocess
from typing import Optional

from fastapi import APIRouter, BackgroundTasks

from server.mlflow_utils import get_project_root

router = APIRouter(prefix="/api", tags=["tasks"])

# Global state for background tasks
TASK_PROCESS: Optional[subprocess.Popen] = None
TASK_LOG_FILE = "daily_task.log"
BACKTEST_PROCESS: Optional[subprocess.Popen] = None
BACKTEST_LOG_FILE = "backtest_task.log"


@router.get("/status")
def get_status():
    """Get status of the daily task."""
    global TASK_PROCESS
    is_running = False
    if TASK_PROCESS:
        ret = TASK_PROCESS.poll()
        if ret is None:
            is_running = True
        else:
            TASK_PROCESS = None  # Cleanup
            
    # Read log tail
    log_content = ""
    if os.path.exists(TASK_LOG_FILE):
        with open(TASK_LOG_FILE, "r") as f:
            try:
                f.seek(0, os.SEEK_END)
                size = f.tell()
                f.seek(max(size - 4000, 0))
                log_content = f.read()
            except Exception:
                pass
            
    return {
        "running": is_running,
        "log": log_content
    }


@router.post("/run-daily")
def run_daily_task(background_tasks: BackgroundTasks):
    """Start the daily trading recommendation task."""
    global TASK_PROCESS
    if TASK_PROCESS and TASK_PROCESS.poll() is None:
        return {"status": "already_running"}
        
    project_root = get_project_root()
    log_path = os.path.join(project_root, TASK_LOG_FILE)
    log_file = open(log_path, "w")
    
    TASK_PROCESS = subprocess.Popen(
        [sys.executable, "-u", "scripts/ops/run_daily_task.py"],
        cwd=project_root, 
        stdout=log_file,
        stderr=subprocess.STDOUT
    )
    return {"status": "started"}


@router.get("/backtest-status")
def get_backtest_status():
    """Get the status and log of the backtest task."""
    global BACKTEST_PROCESS
    is_running = False
    exit_code = None
    
    if BACKTEST_PROCESS:
        ret = BACKTEST_PROCESS.poll()
        if ret is None:
            is_running = True
        else:
            exit_code = ret
            BACKTEST_PROCESS = None  # Cleanup
            
    # Read log tail
    project_root = get_project_root()
    log_path = os.path.join(project_root, BACKTEST_LOG_FILE)
    log_content = ""
    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            try:
                f.seek(0, os.SEEK_END)
                size = f.tell()
                f.seek(max(size - 8000, 0))
                log_content = f.read()
            except Exception:
                pass
            
    return {
        "running": is_running,
        "exit_code": exit_code,
        "log": log_content
    }


@router.post("/run-backtest")
def run_backtest(background_tasks: BackgroundTasks):
    """Start a default backtest using run_etf_analysis.py."""
    global BACKTEST_PROCESS
    
    if BACKTEST_PROCESS and BACKTEST_PROCESS.poll() is None:
        return {"status": "already_running"}
    
    # Import backtest config
    from delorean.config import DEFAULT_BACKTEST_PARAMS
    
    project_root = get_project_root()
    log_path = os.path.join(project_root, BACKTEST_LOG_FILE)
    log_file = open(log_path, "w")
    
    # Build command arguments from config
    cmd = [sys.executable, "-u", "scripts/ops/run_etf_analysis.py"]
    cmd.extend(["--start_time", DEFAULT_BACKTEST_PARAMS["start_time"]])
    cmd.extend(["--train_end_time", DEFAULT_BACKTEST_PARAMS["train_end_time"]])
    cmd.extend(["--test_start_time", DEFAULT_BACKTEST_PARAMS["test_start_time"]])
    cmd.extend(["--topk", str(DEFAULT_BACKTEST_PARAMS.get("topk", 4))])
    cmd.extend(["--label_horizon", str(DEFAULT_BACKTEST_PARAMS.get("label_horizon", 1))])
    cmd.extend(["--smooth_window", str(DEFAULT_BACKTEST_PARAMS.get("smooth_window", 10))])
    
    BACKTEST_PROCESS = subprocess.Popen(
        cmd,
        cwd=project_root, 
        stdout=log_file,
        stderr=subprocess.STDOUT
    )
    return {"status": "started"}
