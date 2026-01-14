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


@router.get("/performance")
def get_performance():
    """Get historical strategy performance for charting."""
    root = get_project_root()
    report_path = os.path.join(root, "artifacts", "backtest_report.pkl")
    
    if not os.path.exists(report_path):
        return {"chart_data": [], "message": "No backtest data available"}
    
    try:
        report = pd.read_pickle(report_path)
        
        # Calculate cumulative returns
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
        
        # Sample for performance (max 200 points)
        if len(chart_data) > 200:
            step = len(chart_data) // 200
            chart_data = chart_data[::step]
        
        return {"chart_data": chart_data}
    except Exception as e:
        return {"chart_data": [], "error": str(e)}


@router.get("/recommendation-history")
def get_recommendation_history():
    """Get last 7 days of recommendation history."""
    import csv
    
    root = get_project_root()
    csv_path = os.path.join(root, "artifacts", "historical_recommendations.csv")
    
    if not os.path.exists(csv_path):
        return {"history": [], "message": "No history available"}
    
    try:
        # Parse the wide-format CSV
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        # Extract dates from column names
        date_cols = [c for c in rows[0].keys() if c.endswith('_name')]
        dates = sorted(set(c.split('_name')[0] for c in date_cols), reverse=True)[:7]
        
        history = []
        for date in dates:
            day_recs = []
            for row in rows[:5]:  # Top 5
                name_col = f"{date}_name"
                symbol_col = f"{date}_symbol"
                score_col = f"{date}_score"
                
                if name_col in row:
                    day_recs.append({
                        "rank": int(row["rank"]),
                        "name": row.get(name_col, ""),
                        "symbol": row.get(symbol_col, ""),
                        "score": float(row.get(score_col, 0))
                    })
            
            history.append({
                "date": date,
                "recommendations": day_recs
            })
        
        return {"history": history}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"history": [], "error": str(e)}
