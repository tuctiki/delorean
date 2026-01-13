import sys
import os
import subprocess
import json
from typing import Optional, List, Dict, Any
import pandas as pd
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Ensure we can import from project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import qlib
    from qlib.data import D
    # Initialize Qlib
    from delorean.config import QLIB_PROVIDER_URI, QLIB_REGION
    qlib.init(provider_uri=QLIB_PROVIDER_URI, region=QLIB_REGION)
    print(f"Qlib initialized with {QLIB_PROVIDER_URI}")
except ImportError:
    print("Warning: Qlib not found or configuration error. detailed data features will be disabled.")
except Exception as e:
    print(f"Warning: Qlib init failed: {e}")

app = FastAPI(title="Delorean Strategy Dashboard")

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all for local dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class Experiment(BaseModel):
    id: str
    name: str
    artifact_location: str

class TradeRecommendation(BaseModel):
    symbol: str
    score: float

class DailyArtifact(BaseModel):
    date: str
    market_status: str
    market_data: Dict[str, float]
    top_recommendations: List[TradeRecommendation]
    full_rankings: List[TradeRecommendation]

# Global state for daily task
TASK_PROCESS: Optional[subprocess.Popen] = None
TASK_LOG_FILE = "daily_task.log"

@app.get("/api/status")
def get_status():
    global TASK_PROCESS
    is_running = False
    if TASK_PROCESS:
        ret = TASK_PROCESS.poll()
        if ret is None:
            is_running = True
        else:
            TASK_PROCESS = None # Cleanup
            
    # Read log tail
    log_content = ""
    if os.path.exists(TASK_LOG_FILE):
        with open(TASK_LOG_FILE, "r") as f:
            # Read last 2000 chars
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

def run_daily_worker():
    # Helper to run the process
    pass

@app.post("/api/run-daily")
def run_daily_task(background_tasks: BackgroundTasks):
    global TASK_PROCESS
    if TASK_PROCESS and TASK_PROCESS.poll() is None:
        return {"status": "already_running"}
        
    # Start process
    # We run 'python scripts/run_daily_task.py' -u for unbuffered
    # Use cwd as project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log_path = os.path.join(project_root, TASK_LOG_FILE)
    log_file = open(log_path, "w")
    
    # Use standard python command
    TASK_PROCESS = subprocess.Popen(
        [sys.executable, "-u", "scripts/run_daily_task.py"],
        cwd=project_root, 
        stdout=log_file,
        stderr=subprocess.STDOUT
    )
    return {"status": "started"}

@app.get("/api/recommendations")
def get_recommendations():
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "daily_recommendations.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}

@app.get("/api/experiments")
def list_experiments():
    """List all experiments with summary metrics from latest run."""
    experiments = []
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    mlruns_path = os.path.join(root, "mlruns")
    
    if os.path.exists(mlruns_path):
        for d in os.listdir(mlruns_path):
            full_path = os.path.join(mlruns_path, d)
            if os.path.isdir(full_path) and d.isdigit():
                # Get creation time
                creation_time = os.path.getmtime(full_path)
                creation_str = pd.Timestamp(creation_time, unit='s').strftime('%Y-%m-%d %H:%M:%S')
                 
                # Read meta.yaml for experiment name
                exp_name = f"Experiment {d}"
                meta_path = os.path.join(full_path, "meta.yaml")
                if os.path.exists(meta_path):
                    try:
                        with open(meta_path, "r") as f:
                            for line in f:
                                if line.strip().startswith("name:"):
                                    exp_name = line.split(":", 1)[1].strip()
                                    break
                    except: pass
                
                # [NEW] Read metrics from latest run
                metrics = {}
                runs = [r for r in os.listdir(full_path) if os.path.isdir(os.path.join(full_path, r)) and len(r) > 20]
                if runs:
                    runs.sort(key=lambda x: os.path.getmtime(os.path.join(full_path, x)), reverse=True)
                    latest_run = runs[0]
                    metrics_path = os.path.join(full_path, latest_run, "metrics")
                    if os.path.exists(metrics_path):
                        for mf in os.listdir(metrics_path):
                            try:
                                with open(os.path.join(metrics_path, mf), "r") as f:
                                    # MLflow format: timestamp value step
                                    parts = f.read().strip().split()
                                    if len(parts) >= 2:
                                        metrics[mf] = float(parts[1])  # Value is second column
                            except: pass
                 
                experiments.append({
                    "id": d, 
                    "name": exp_name, 
                    "artifact_location": full_path,
                    "creation_time": creation_str,
                    "timestamp": creation_time,
                    "metrics": metrics
                })
    
    # Sort by Timestamp Descending
    experiments.sort(key=lambda x: x["timestamp"], reverse=True)
    return experiments

@app.get("/api/experiment_results")
def get_experiment_results():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, "artifacts", "experiment_results.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}

@app.get("/api/experiments/{experiment_id}")
def get_experiment_details(experiment_id: str):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    exp_path = os.path.join(root, "mlruns", experiment_id)
    
    if not os.path.exists(exp_path):
        raise HTTPException(status_code=404, detail="Experiment not found")
        
    details = {
        "id": experiment_id,
        "artifact_location": exp_path,
        "params": {},
        "metrics": {},
        "status": "FINISHED" 
    }
    
    # Simple parse of mlruns structure (assuming FileStore)
    # usually there's a meta.yaml and subdirs for runs.
    # But Qlib's default FileSystemRecorder structure:
    # mlruns/experiment_id/run_id/params/
    # mlruns/experiment_id/run_id/metrics/
    
    # We will just pick the LATEST run for this experiment to show details
    if os.path.exists(exp_path):
        runs = [d for d in os.listdir(exp_path) if os.path.isdir(os.path.join(exp_path, d)) and len(d) > 20]
        # Sort by modification time to get latest
        runs.sort(key=lambda x: os.path.getmtime(os.path.join(exp_path, x)), reverse=True)
        
        if runs:
            latest_run = runs[0]
            run_path = os.path.join(exp_path, latest_run)
            
            # Read Params
            params_path = os.path.join(run_path, "params")
            if os.path.exists(params_path):
                for f in os.listdir(params_path):
                    try:
                        with open(os.path.join(params_path, f), "r") as pf:
                            details["params"][f] = pf.read().strip()
                    except: pass
                    
            # Read Metrics
            metrics_path = os.path.join(run_path, "metrics")
            if os.path.exists(metrics_path):
                for f in os.listdir(metrics_path):
                    try:
                        with open(os.path.join(metrics_path, f), "r") as mf:
                            # Metrics file contains value timestamp
                            # 0.0023 123123123
                            content = mf.read().split()[0]
                            details["metrics"][f] = float(content)
                    except: pass
                    
    return details

@app.get("/api/experiments/{experiment_id}/image")
def get_experiment_image(experiment_id: str, name: str):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    exp_path = os.path.join(root, "mlruns", experiment_id)
    
    if not os.path.exists(exp_path):
        raise HTTPException(status_code=404, detail="Experiment not found")
        
    # Get latest run
    runs = [d for d in os.listdir(exp_path) if os.path.isdir(os.path.join(exp_path, d)) and len(d) > 20]
    if not runs:
        raise HTTPException(status_code=404, detail="No runs found")
        
    runs.sort(key=lambda x: os.path.getmtime(os.path.join(exp_path, x)), reverse=True)
    latest_run = runs[0]
    
    # Check standard artifacts location (Qlib/MLflow default)
    # 1. mlruns/exp/run/artifacts/name
    # 2. mlruns/exp/run/name (sometimes direct)
    
    possible_paths = [
        os.path.join(exp_path, latest_run, "artifacts", name),
        os.path.join(exp_path, latest_run, name),
        # [REMOVED] Fallback to project artifacts is dangerous as it causes identical plots across experiments
        # os.path.join(root, "artifacts", name)
    ]
    
    for p in possible_paths:
        if os.path.exists(p):
            from fastapi.responses import FileResponse
            return FileResponse(p)
            
    raise HTTPException(status_code=404, detail="Image not found")

@app.get("/api/data/{symbol}")
def get_data(symbol: str, days: int = 365):
    try:
        # Load from D
        fields = ["$close", "$volume"]
        # Fetch larger history then slice
        df = D.features([symbol], fields, start_time="2020-01-01")
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {symbol}")
            
        # Reset index to json
        # Index is MultiIndex (instrument, datetime)
        if hasattr(df.index, "levels"):
             df = df.droplevel(0) # Drop instrument level if present
             
        df = df.reset_index()
        # Ensure columns
        # df columns: [datetime, $close, $volume]
        
        records = []
        for _, row in df.iterrows():
            if pd.isna(row["datetime"]): continue
            records.append({
                "date": row["datetime"].strftime("%Y-%m-%d"),
                "close": row["$close"] if not pd.isna(row["$close"]) else None,
                "volume": row["$volume"] if not pd.isna(row["$volume"]) else None
            })
            
        return records
    except Exception as e:
         import traceback
         traceback.print_exc()
         raise HTTPException(status_code=500, detail=str(e))
         
@app.get("/api/search")
def search_etfs():
    # Return ETF List
    from delorean.config import ETF_LIST
    return ETF_LIST

@app.get("/api/config")
def get_config():
    """Expose current system configuration."""
    from delorean.config import MODEL_PARAMS_STAGE1, MODEL_PARAMS_STAGE2, ETF_LIST, START_TIME, END_TIME
    from delorean.data import ETFDataHandler
    
    custom_exprs, custom_names = ETFDataHandler.get_custom_factors()
    
    return {
        "model_params": {
            "stage1": MODEL_PARAMS_STAGE1,
            "stage2": MODEL_PARAMS_STAGE2
        },
        "data_factors": {
            "names": custom_names,
            "expressions": custom_exprs
        },
        "universe": ETF_LIST,
        "time_range": {
            "start": START_TIME,
            "end": END_TIME
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
