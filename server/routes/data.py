"""
Data API routes.
Handles market data, ETF search, config, and recommendations.
"""

import os
import json
import pandas as pd
from fastapi import APIRouter, HTTPException

from server.mlflow_utils import get_project_root

router = APIRouter(prefix="/api", tags=["data"])


@router.get("/recommendations")
def get_recommendations():
    """Get daily trading recommendations."""
    path = os.path.join(get_project_root(), "daily_recommendations.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


@router.get("/data/{symbol}")
def get_data(symbol: str, days: int = 365):
    """Get historical price data for a symbol."""
    try:
        from qlib.data import D
        
        # Load from D
        fields = ["$close", "$volume"]
        df = D.features([symbol], fields, start_time="2020-01-01")
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {symbol}")
            
        # Reset index to json
        if hasattr(df.index, "levels"):
            df = df.droplevel(0)
             
        df = df.reset_index()
        
        records = []
        for _, row in df.iterrows():
            if pd.isna(row["datetime"]):
                continue
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


@router.get("/search")
def search_etfs():
    """Return ETF list for search."""
    from delorean.config import ETF_LIST
    return ETF_LIST


@router.get("/config")
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
