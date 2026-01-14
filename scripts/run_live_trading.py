"""
Live Trading Signal Generator for ETF Strategy.

Generates daily trading recommendations with two-phase approach:
1. Out-of-sample validation (recent 60 days)
2. Production signal generation (full history)
"""

import qlib
import pandas as pd
import datetime
import sys
import os
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qlib.workflow import R
from qlib.data import D
from qlib.contrib.evaluate import risk_analysis
import matplotlib.pyplot as plt

from delorean.config import (
    QLIB_PROVIDER_URI, QLIB_REGION, BENCHMARK, ETF_NAME_MAP,
    ETF_LIST, LIVE_TRADING_CONFIG
)
from delorean.data import ETFDataLoader
from delorean.model import ModelTrainer
from delorean.backtest import BacktestEngine
from delorean.signals import smooth_predictions
from delorean.regime import get_current_regime, get_regime_status_string


def run_validation(today: datetime.datetime, config: dict) -> dict:
    """
    Phase 1: Out-of-sample validation on recent data.
    
    Args:
        today: Current date.
        config: Configuration dict with validation_days, label_horizon, etc.
        
    Returns:
        Dictionary with validation metrics (rank_ic, sharpe, status).
    """
    print("\n[Phase 1/2] Out-of-Sample Validation...")
    
    val_days = config["validation_days"]
    label_horizon = config["label_horizon"]
    topk = config["topk"]
    
    val_split_date = today - datetime.timedelta(days=val_days)
    val_train_end = (val_split_date - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
    val_test_start = val_split_date.strftime("%Y-%m-%d")
    
    print(f"  > Training Data: ... to {val_train_end}")
    print(f"  > Test (Validation) Data: {val_test_start} to Present")
    
    # Load and train
    data_loader = ETFDataLoader(label_horizon=label_horizon)
    dataset = data_loader.load_data(train_end=val_train_end, test_start=val_test_start)
    
    model = ModelTrainer()
    model.train(dataset)
    pred_scores = model.predict(dataset)
    
    # Calculate metrics
    validation_metrics = {
        "rank_ic": 0.0,
        "sharpe": 0.0,
        "ic_status": "Unknown",
        "sharpe_status": "Unknown"
    }
    
    df_val = dataset.prepare("test", col_set=["label"])
    y_val = df_val["label"]
    val_res = pd.DataFrame({"score": pred_scores, "label": y_val.iloc[:, 0]}).dropna()
    
    if val_res.empty:
        print("  Warning: No validation data available.")
        return validation_metrics
    
    # Rank IC
    daily_ic = val_res.groupby(level='datetime').apply(
        lambda x: x["score"].corr(x["label"], method="spearman")
    )
    mean_ic = daily_ic.mean()
    validation_metrics["rank_ic"] = float(mean_ic)
    print(f"  Validation Rank IC: {mean_ic:.4f}")
    
    validation_metrics["ic_status"] = "Pass" if mean_ic >= 0.02 else "Warning"
    
    # Sharpe via backtest
    try:
        engine = BacktestEngine(pred_scores)
        report, _ = engine.run(topk=topk)
        risks = risk_analysis(report["return"], freq="day")
        sharpe = risks.loc["information_ratio", "risk"]
        validation_metrics["sharpe"] = float(sharpe)
        print(f"  Validation Sharpe: {sharpe:.4f}")
        
        if sharpe < 0.0:
            validation_metrics["sharpe_status"] = "Critical"
        elif sharpe < 0.5:
            validation_metrics["sharpe_status"] = "Warning"
        else:
            validation_metrics["sharpe_status"] = "Pass"
            
        # Save validation plot
        cum_ret = (1 + report["return"]).cumprod()
        plt.figure(figsize=(10, 5))
        plt.plot(cum_ret.index, cum_ret.values, label="Strategy", color="#58a6ff")
        plt.title(f"Validation Performance ({val_test_start} - Present)")
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.legend()
        plt.savefig("artifacts/validation_plot.png")
        plt.close()
        
    except Exception as e:
        print(f"  Warning: Sharpe calculation failed: {e}")
        validation_metrics["sharpe_status"] = "Error"
    
    # Log to MLflow
    _log_validation_metrics(validation_metrics, model)
    
    return validation_metrics


def _log_validation_metrics(metrics: dict, model: ModelTrainer) -> None:
    """Log validation metrics to MLflow."""
    import mlflow
    if mlflow.active_run():
        mlflow.end_run()
        
    with R.start(experiment_name="daily_validation"):
        R.log_params(**model.get_params())
        R.log_metrics(rank_ic=metrics["rank_ic"], sharpe=metrics["sharpe"])
        
        plot_path = "artifacts/validation_plot.png"
        if os.path.exists(plot_path):
            R.log_artifact(plot_path, "cumulative_return.png")
        print("  [MLflow] Metrics logged to 'daily_validation'.")


def generate_production_signal(today: datetime.datetime, config: dict) -> pd.Series:
    """
    Phase 2: Generate production trading signals.
    
    Args:
        today: Current date.
        config: Configuration dict with production_split_days, smooth_window, etc.
        
    Returns:
        Smoothed prediction series for latest date.
    """
    print("\n[Phase 2/2] Production Signal Generation...")
    
    prod_split_days = config["production_split_days"]
    label_horizon = config["label_horizon"]
    smooth_window = config["smooth_window"]
    
    prod_split_date = today - datetime.timedelta(days=prod_split_days)
    prod_train_end = (prod_split_date - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
    prod_test_start = prod_split_date.strftime("%Y-%m-%d")
    
    print(f"  > Training Data: ... to {prod_train_end}")
    print(f"  > Prediction Target: {prod_test_start} to Present")
    
    # Load and train
    data_loader = ETFDataLoader(label_horizon=label_horizon)
    dataset = data_loader.load_data(train_end=prod_train_end, test_start=prod_test_start)
    
    model = ModelTrainer()
    model.train(dataset)
    pred = model.predict(dataset)
    
    # Smooth predictions
    print(f"  > Applying {smooth_window}-day EWMA Smoothing...")
    pred_smooth = smooth_predictions(pred, halflife=smooth_window)
    
    return pred_smooth


def build_recommendation_artifact(
    pred: pd.Series,
    is_bull: bool,
    market_data: dict,
    validation_metrics: dict,
    config: dict
) -> dict:
    """
    Build the recommendation JSON artifact.
    
    Args:
        pred: Smoothed predictions for latest date.
        is_bull: Bull market status.
        market_data: Market regime data (close, ma).
        validation_metrics: Validation phase results.
        config: Configuration dict.
        
    Returns:
        Complete recommendation artifact dict.
    """
    latest_date = pred.index.get_level_values('datetime').max()
    latest_pred = pred.loc[latest_date].sort_values(ascending=False)
    
    topk = config["topk"]
    buffer_size = config["buffer_size"]
    smooth_window = config["smooth_window"]
    label_horizon = config["label_horizon"]
    
    # Strategy config section
    strategy_config = {
        "topk": topk,
        "smooth_window": smooth_window,
        "buffer": buffer_size,
        "label_horizon": label_horizon,
        "mode": "Equal Weight + Global HS300 Filter"
    }
    
    artifact = {
        "date": latest_date.strftime('%Y-%m-%d'),
        "generation_time": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "market_status": get_regime_status_string(is_bull),
        "validation": validation_metrics,
        "market_data": market_data,
        "strategy_config": strategy_config,
        "top_recommendations": [],
        "buffer_holdings": [],
        "full_rankings": []
    }
    
    # Fetch additional data for recommendations
    fields = ['$close', 'Std($close/Ref($close,1)-1, 20)']
    feat_df = D.features(ETF_LIST, fields, start_time=latest_date, end_time=latest_date)
    
    vol_map, close_map = {}, {}
    if not feat_df.empty:
        feat_df.columns = ['close', 'vol20']
        if 'datetime' in feat_df.index.names:
            feat_df = feat_df.droplevel('datetime')
        vol_map = feat_df['vol20'].to_dict()
        close_map = feat_df['close'].to_dict()
    
    # Populate recommendations
    if is_bull:
        for i, (symbol, score) in enumerate(latest_pred.head(topk + buffer_size).items(), 1):
            is_buffer = i > topk
            weight = 1.0 / topk if not is_buffer else 0.0
            
            artifact["top_recommendations"].append({
                "rank": i,
                "symbol": symbol,
                "name": ETF_NAME_MAP.get(symbol, symbol),
                "score": float(score),
                "volatility": float(vol_map.get(symbol, 0.0)),
                "current_price": float(close_map.get(symbol, 0.0)),
                "target_weight": float(weight),
                "is_buffer": is_buffer
            })
    
    # Full rankings
    for symbol, score in latest_pred.items():
        artifact["full_rankings"].append({
            "symbol": symbol,
            "score": float(score),
            "volatility": float(vol_map.get(symbol, 0.0))
        })
    
    return artifact


def print_recommendations(pred: pd.Series, config: dict, is_bull: bool) -> None:
    """Print formatted recommendations to console."""
    latest_date = pred.index.get_level_values('datetime').max()
    latest_pred = pred.loc[latest_date].sort_values(ascending=False)
    
    topk = config["topk"]
    buffer_size = config["buffer_size"]
    
    print(f"\n[Result] Latest Signal Date: {latest_date.strftime('%Y-%m-%d')}")
    
    print("\n" + "-"*30)
    print(f"  Top {topk} Recommendations")
    print("-" * 30)
    for i, (symbol, score) in enumerate(latest_pred.head(topk).items(), 1):
        print(f"  #{i}  {symbol:<10} (Score: {score:.4f})")
    print("-" * 30)
    
    print("\nFull Rankings (for manual turnover check):")
    print(latest_pred)
    
    print("\n[Strategy Note]")
    print(f"- Target Hold: Top {topk}")
    print(f"- Buffer Logic (Hysteresis): Keep existing holdings if Rank <= {topk + buffer_size}.")
    print(f"- Turnover Control: Only swap if Rank > {topk + buffer_size}.")
    print("- Regime Filter: If Bear Market Warning above, prefer CASH.")


def get_trading_signal(topk: int = None) -> None:
    """
    Main orchestrator: Generate trading signals for the latest date.
    
    Args:
        topk: Number of top ETFs. Defaults to config value.
    """
    # Initialize
    qlib.init(provider_uri=QLIB_PROVIDER_URI, region=QLIB_REGION)
    
    config = LIVE_TRADING_CONFIG.copy()
    if topk is not None:
        config["topk"] = topk
    
    print("\n" + "="*50)
    print("  ETF Strategy Live Signal Generator")
    print("="*50)
    
    today = datetime.datetime.now()
    
    # Phase 1: Validation
    validation_metrics = run_validation(today, config)
    
    # Phase 2: Production signals
    pred_smooth = generate_production_signal(today, config)
    
    # Market regime check
    latest_date = pred_smooth.index.get_level_values('datetime').max()
    print("\n[Market Regime Check] Global HS300 Filter...")
    
    is_bull = True # Default to Bull (Always Trade)
    market_data = {
        "benchmark_close": 0.0, 
        "benchmark_ma": 0.0, 
        "ma_window": config.get("regime_ma_window", 60)
    }

    if config.get("use_regime_filter", True):
        is_bull, market_data = get_current_regime(latest_date)
        status = "BULL" if is_bull else "BEAR"
        print(f"  [Status: ENABLED] Result: {status} (Close: {market_data['benchmark_close']:.2f}, MA{market_data['ma_window']}: {market_data['benchmark_ma']:.2f})")
    else:
        # Still fetch data for reporting if possible, but force Bull
        try:
             _, real_market_data = get_current_regime(latest_date)
             market_data = real_market_data
        except Exception as e:
             pass
        print(f"  [Status: DISABLED] Forcing BULL market state (Close: {market_data.get('benchmark_close', 0.0):.2f})")
    
    # Build and save artifact
    artifact = build_recommendation_artifact(
        pred_smooth, is_bull, market_data, validation_metrics, config
    )
    
    with open("daily_recommendations.json", "w") as f:
        json.dump(artifact, f, indent=2)
    print("\n[Artifact] Saved to 'daily_recommendations.json'")
    
    # Print results
    print_recommendations(pred_smooth, config, is_bull)


if __name__ == "__main__":
    get_trading_signal()
