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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from qlib.workflow import R
from qlib.data import D
from qlib.contrib.evaluate import risk_analysis
import matplotlib.pyplot as plt

from delorean.config import (
    QLIB_PROVIDER_URI, QLIB_REGION, BENCHMARK, REGIME_BENCHMARK, ETF_NAME_MAP,
    ETF_LIST, LIVE_TRADING_CONFIG
)
from delorean.data import ETFDataLoader
from delorean.model import ModelTrainer
from delorean.backtest import BacktestEngine
from delorean.signals import smooth_predictions
from delorean.strategy.portfolio import PortfolioOptimizer


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
    # Define segments for DoubleEnsemble stability
    val_train_start = "2015-01-01"
    val_valid_start = (val_split_date - datetime.timedelta(days=61)).strftime("%Y-%m-%d")
    val_valid_end = (val_split_date - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
    val_train_end = (val_split_date - datetime.timedelta(days=62)).strftime("%Y-%m-%d")
    val_test_start = val_split_date.strftime("%Y-%m-%d")
    
    # Load and train
    data_loader = ETFDataLoader(label_horizon=label_horizon)
    dataset = data_loader.load_data(
        train_start=val_train_start,
        train_end=val_train_end,
        valid_start=val_valid_start,
        valid_end=val_valid_end,
        test_start=val_test_start
    )
    
    model = ModelTrainer()
    model.train(dataset, model_type="double_ensemble")
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
        from delorean.utils import run_standard_backtest
        report, _ = run_standard_backtest(
            pred=pred_scores,
            topk=topk,
            buffer=config.get("buffer_size", 2),
            target_vol=config.get("target_vol", 0.20),
            use_regime_filter=True,
            use_trend_filter=False,
            n_drop=config.get("n_drop", 2),
            rebalance_threshold=config.get("rebalance_threshold", 0.05)
        )
        risks = risk_analysis(report["return"], freq="day")
        sharpe = risks.loc["information_ratio", "risk"]
        validation_metrics["sharpe"] = float(sharpe)
        print(f"  Validation Sharpe Ratio: {sharpe:.4f}")
        
        if sharpe < 0.0:
            validation_metrics["sharpe_status"] = "Critical"
        elif sharpe < 0.4:
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
    prod_train_start = "2015-01-01"
    prod_valid_start = (prod_split_date - datetime.timedelta(days=61)).strftime("%Y-%m-%d")
    prod_valid_end = (prod_split_date - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
    prod_train_end = (prod_split_date - datetime.timedelta(days=62)).strftime("%Y-%m-%d")
    prod_test_start = prod_split_date.strftime("%Y-%m-%d")
    
    # Load and train
    data_loader = ETFDataLoader(label_horizon=label_horizon)
    dataset = data_loader.load_data(
        train_start=prod_train_start,
        train_end=prod_train_end,
        valid_start=prod_valid_start,
        valid_end=prod_valid_end,
        test_start=prod_test_start
    )
    
    model = ModelTrainer()
    model.train(dataset, model_type="double_ensemble")
    pred = model.predict(dataset)
    
    # Smooth predictions

    print(f"  > Applying {smooth_window}-day EWMA Smoothing...")
    pred_smooth = smooth_predictions(pred, halflife=smooth_window)
    
    return pred_smooth


def build_recommendation_artifact(
    pred: pd.Series,
    regime_ratio: float,
    metrics_ic: float,
    validation_metrics: dict,
    config: dict
) -> dict:
    """
    Build the recommendation JSON artifact using unified PortfolioOptimizer.
    
    Args:
        pred: Smoothed predictions for latest date.
        regime_ratio: Market regime ratio (Price/MA60).
        metrics_ic: Rank IC for the latest period.
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
    target_vol_config = config.get("target_vol", 0.20)
    
    # 1. Initialize Unified Portfolio Optimizer
    # Note: risk_degree=0.95 is standard for the strategy
    optimizer = PortfolioOptimizer(topk=topk, risk_degree=0.95)
    
    # 2. Fetch Volatility and Benchmark Data for Weighting
    # We need VOL20 for Risk Parity and Scaling
    from delorean.utils import fetch_volatility_feature
    vol_feature = fetch_volatility_feature(ETF_LIST, latest_date, latest_date)
    
    close_map = {}
    if not vol_feature.empty:
        # Also extract close prices from the same fetch
        fields = ['$close']
        close_df = D.features(ETF_LIST, fields, start_time=latest_date, end_time=latest_date)
        if not close_df.empty:
            close_map = close_df.iloc[:, 0].reset_index().set_index('instrument')['$close'].to_dict()

    # 3. Calculate Unified Weights
    target_stocks = latest_pred.head(topk).index.tolist()
    target_weights = optimizer.calculate_weights(
        target_stocks=target_stocks,
        current_date=latest_date,
        vol_feature=vol_feature,
        target_vol=target_vol_config,
        regime_ratio=regime_ratio
    )
    
    # 4. Identify Market Status (Consistent with Optimizer)
    from delorean.strategy.portfolio import BULL_THRESHOLD, BEAR_THRESHOLD
    if regime_ratio >= BULL_THRESHOLD:
        market_status = "STREET BULL (Risk On)"
    elif regime_ratio <= BEAR_THRESHOLD:
        market_status = "BEAR MARKET (Risk Off)"
    else:
        market_status = f"NEUTRAL/TRANSITION ({regime_ratio:.2f})"

    # 5. Build Artifact
    strategy_config = {
        "topk": topk,
        "smooth_window": smooth_window,
        "buffer": buffer_size,
        "label_horizon": label_horizon,
        "target_vol": target_vol_config,
        "n_drop": config.get("n_drop", 2),
        "rebalance_threshold": config.get("rebalance_threshold", 0.05),
        "risk_degree": 0.95,
        "mode": "Risk Parity + Asymmetric Vol Scaling"
    }
    
    artifact = {
        "date": latest_date.strftime('%Y-%m-%d'),
        "generation_time": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "market_status": market_status,
        "regime_ratio": float(regime_ratio),
        "validation": validation_metrics,
        "strategy_config": strategy_config,
        "top_recommendations": [],
        "buffer_holdings": [],
        "full_rankings": []
    }
    
    # Calculate additional metrics for display
    # (Estimated Volatility of the top-k components)
    topk_vols = []
    for s in target_stocks:
        try:
            v = vol_feature.loc[(latest_date, s)]
            if not pd.isna(v): topk_vols.append(v)
        except: continue
    
    avg_vol_topk = sum(topk_vols) / len(topk_vols) if topk_vols else 0.0
    ann_vol_est = avg_vol_topk * (252 ** 0.5)
    artifact["estimated_ann_vol"] = float(ann_vol_est)

    # Populate recommendations
    # We show top-k + buffer for the user to see hysteresis candidates
    for i, (symbol, score) in enumerate(latest_pred.head(topk + buffer_size).items(), 1):
        is_buffer = i > topk
        
        # Get weight from optimizer (will be 0 for buffer stocks if not in target_stocks)
        weight = target_weights.get(symbol, 0.0)
        
        artifact["top_recommendations"].append({
            "rank": i,
            "symbol": symbol,
            "name": ETF_NAME_MAP.get(symbol, symbol),
            "score": float(score),
            "volatility": float(vol_feature.loc[(latest_date, symbol)]) if vol_feature is not None and (latest_date, symbol) in vol_feature.index else 0.0,
            "current_price": float(close_map.get(symbol, 0.0)),
            "target_weight": float(weight),
            "is_buffer": is_buffer
        })
    
    # Full rankings
    for symbol, score in latest_pred.items():
        artifact["full_rankings"].append({
            "symbol": symbol,
            "score": float(score),
            "volatility": float(vol_feature.loc[(latest_date, symbol)]) if vol_feature is not None and (latest_date, symbol) in vol_feature.index else 0.0
        })
    
    return artifact


def print_recommendations(pred: pd.Series, config: dict, regime_ratio: float) -> None:
    """Print formatted recommendations to console."""
    latest_date = pred.index.get_level_values('datetime').max()
    latest_pred = pred.loc[latest_date].sort_values(ascending=False)
    
    topk = config["topk"]
    buffer_size = config["buffer_size"]
    
    print(f"\n[Result] Latest Signal Date: {latest_date.strftime('%Y-%m-%d')}")
    print(f"[Market] Regime Ratio: {regime_ratio:.2f} ({'BULL' if regime_ratio >= 1.0 else 'BEAR' if regime_ratio <= 0.0 else 'TRANSITION'})")
    
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
    print("- Turnover Control: Only swap if Rank > {topk + buffer_size}.")
    print(f"- Portfolio Mode: Asymmetric Vol Scaling (Current Ratio: {regime_ratio:.2f})")


def get_trading_signal(topk: int = None, n_drop: int = None, rebalance_threshold: float = None) -> None:
    """
    Main orchestrator: Generate trading signals for the latest date.
    
    Args:
        topk: Number of top ETFs. Defaults to config value.
        n_drop: Max ETFs to drop. Defaults to config.
        rebalance_threshold: Portfolio turnover threshold.
    """
    # Initialize
    qlib.init(provider_uri=QLIB_PROVIDER_URI, region=QLIB_REGION)
    
    config = LIVE_TRADING_CONFIG.copy()
    if topk is not None:
        config["topk"] = topk
    if n_drop is not None:
        config["n_drop"] = n_drop
    if rebalance_threshold is not None:
        config["rebalance_threshold"] = rebalance_threshold
    
    print("\n" + "="*50)
    print("  ETF Strategy Live Signal Generator")
    print("="*50)
    
    today = datetime.datetime.now()
    
    # Phase 1: Validation
    validation_metrics = run_validation(today, config)
    
    # Phase 2: Production signals
    pred_smooth = generate_production_signal(today, config)
    
    # Market status (CSI300 / MA60 Ratio)
    print("\n[Market Status] Calculating Regime Ratio...")
    from delorean.utils import fetch_regime_ratio
    try:
        # Fetch last 120 days to ensure we have MA60 data
        regime_series = fetch_regime_ratio(
            REGIME_BENCHMARK,
            (today - datetime.timedelta(days=120)).strftime("%Y-%m-%d"),
            today.strftime("%Y-%m-%d")
        )
        
        if regime_series.empty:
            print("  Warning: No benchmark data for regime filter. Defaulting to BULL.")
            regime_ratio = 1.01
            benchmark_close = None
            benchmark_ma60 = None
        else:
            regime_ratio = regime_series.iloc[-1]
            print(f"  > CSI300 Price/MA60 Ratio: {regime_ratio:.4f}")
            # Extract benchmark price and MA60 for dashboard
            try:
                from qlib.data import D
                bench_df = D.features([BENCHMARK], ["$close"], 
                                     start_time=(today - datetime.timedelta(days=120)).strftime("%Y-%m-%d"),
                                     end_time=today.strftime("%Y-%m-%d"))
                if not bench_df.empty:
                    bench_series = bench_df.droplevel(0)['$close']
                    benchmark_close = float(bench_series.iloc[-1])
                    benchmark_ma60 = float(bench_series.rolling(60).mean().iloc[-1])
                else:
                    benchmark_close = None
                    benchmark_ma60 = None
            except:
                benchmark_close = None
                benchmark_ma60 = None
    except Exception as e:
        print(f"  Warning: Regime filter calculation failed: {e}. Defaulting to BULL.")
        regime_ratio = 1.01
        benchmark_close = None
        benchmark_ma60 = None
    
    # Build and save artifact
    artifact = build_recommendation_artifact(
        pred_smooth, regime_ratio, 0.0, validation_metrics, config
    )
    
    # Add market data for dashboard
    artifact["market_data"] = {
        "benchmark_close": benchmark_close,
        "benchmark_ma60": benchmark_ma60
    }
    
    with open("daily_recommendations.json", "w") as f:
        json.dump(artifact, f, indent=2)
    print("\n[Artifact] Saved to 'daily_recommendations.json'")
    
    # Print results
    print_recommendations(pred_smooth, config, regime_ratio)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--topk", type=int, help="Override TopK")
    parser.add_argument("--n_drop", type=int, help="Override n_drop")
    parser.add_argument("--rebalance_threshold", type=float, help="Override rebalance_threshold")
    args = parser.parse_args()
    
    get_trading_signal(topk=args.topk, n_drop=args.n_drop, rebalance_threshold=args.rebalance_threshold)
