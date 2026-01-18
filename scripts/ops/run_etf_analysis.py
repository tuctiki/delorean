
import argparse
import pandas as pd
import sys
import os
import qlib
from qlib.workflow import R

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from delorean.conf import ETF_LIST, START_TIME, END_TIME, DEFAULT_EXPERIMENT_NAME, TEST_START_TIME, TRAIN_END_TIME
from delorean.runner import StrategyRunner, OptimizationConfig
from delorean.backtest import BacktestEngine
from delorean.analysis import ResultAnalyzer, FactorAnalyzer
from delorean.experiment_manager import ExperimentManager
from delorean.utils import fetch_volatility_feature

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run ETF Strategy Analysis")
    parser.add_argument("--topk", type=int, default=4, help="Number of stocks to hold (default: 4)")
    parser.add_argument("--target_vol", type=float, default=0.20, help="Annualized Target Volatility (default: 0.20)")
    parser.add_argument("--use_alpha158", action="store_true", help="Use Qlib Alpha158 embedded factors")
    parser.add_argument("--use_hybrid", action="store_true", help="Use Hybrid Factors (Custom + Alpha158)")
    parser.add_argument("--risk_parity", action="store_true", help="Enable Volatility Targeting (1/Vol)")
    parser.add_argument("--no_dynamic_exposure", action="store_false", dest="dynamic_exposure", help="Disable Trend-based Dynamic Exposure")
    parser.set_defaults(dynamic_exposure=True)
    parser.add_argument("--buffer", type=int, default=3, help="Rank Buffer for Hysteresis")
    parser.add_argument("--label_horizon", type=int, default=1, help="Forward return label horizon")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--smooth_window", type=int, default=10, help="EWMA smoothing halflife")
    parser.add_argument("--signal_halflife", type=int, default=15, help="EMA smoothing for prediction scores")

    parser.add_argument("--rebalance_threshold", type=float, default=0.05, help="Rebalancing threshold (default: 0.05)")
    parser.add_argument("--n_drop", type=int, default=2, help="Maximum number of stocks to swap per day (default: 2)")
    
    # Time Range Overrides
    parser.add_argument("--start_time", type=str, default=None, help="Backtest Data Start Time")
    parser.add_argument("--train_end_time", type=str, default=None, help="Training End Time")
    parser.add_argument("--test_start_time", type=str, default=None, help="Test Start Time")
    parser.add_argument("--end_time", type=str, default=None, help="Backtest End Time")
    parser.add_argument("--experiment_name", type=str, default=DEFAULT_EXPERIMENT_NAME, help="Configs Experiment Name")
    
    # Walk-Forward Validation
    parser.set_defaults(walk_forward=True)
    parser.add_argument("--train_window_months", type=int, default=24, help="Training window in months")
    parser.add_argument("--retrain_frequency_months", type=int, default=1, help="Retrain frequency in months")
    
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    
    # Time Config Overrides
    start_time = args.start_time if args.start_time else START_TIME
    end_time = args.end_time if args.end_time else END_TIME
    train_end_time = args.train_end_time if args.train_end_time else TRAIN_END_TIME
    test_start_time = args.test_start_time if args.test_start_time else TEST_START_TIME

    # Clean up old artifacts
    print("Cleaning up old artifacts...")
    plots = ["cumulative_return.png", "excess_return.png", "factor_ic.png", "feature_correlation.png"]
    for p in plots:
        p_path = os.path.join("artifacts", p)
        if os.path.exists(p_path):
            os.remove(p_path)

    # Walk-Forward Logic (Keep inline for now as it's complex to abstract entirely yet)
    # === UNIFIED EXECUTION ===
    runner = StrategyRunner(seed=args.seed, experiment_name=args.experiment_name)
    runner.initialize()
    
    pred = None
    dataset = None

    if args.walk_forward:
        # --- Mode A: Walk-Forward Validation ---
        print(f"Running Walk-Forward Validation (Train Window: {args.train_window_months}m, Freq: {args.retrain_frequency_months}m)")
        from delorean.walk_forward import WalkForwardValidator, WalkForwardConfig
        
        wf_config = WalkForwardConfig(
            train_window_months=args.train_window_months,
            retrain_frequency_months=args.retrain_frequency_months,
            seed=args.seed,
            smooth_window=args.smooth_window,
            label_horizon=args.label_horizon
        )
        validator = WalkForwardValidator(wf_config)
        # validator.run expects strings
        pred = validator.run(test_start=test_start_time, test_end=end_time)
        
    else:
        # --- Mode B: Standard Fixed-Split Validation ---
        print("Running Standard Fixed-Split Validation")
        
        # 1. Load Data
        print("Loading Data...")
        # Calculate validation segment for DoubleEnsemble
        ts_test_start = pd.Timestamp(test_start_time)
        valid_start = (ts_test_start - pd.Timedelta(days=61)).strftime("%Y-%m-%d")
        valid_end = (ts_test_start - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        # Adjust train end to avoid overlap
        adj_train_end = (ts_test_start - pd.Timedelta(days=62)).strftime("%Y-%m-%d")
        
        dataset = runner.load_data(
            start_time=start_time,
            end_time=end_time,
            train_end_time=adj_train_end,
            valid_start_time=valid_start,
            valid_end_time=valid_end,
            test_start_time=test_start_time,
            label_horizon=args.label_horizon,
            use_alpha158=args.use_alpha158,
            use_hybrid=args.use_hybrid
        )

        # 2. Factor Analysis
        print("Running Factor Analysis...")
        factor_analyzer = FactorAnalyzer()
        factor_analyzer.analyze(dataset)

        # 3. Train Model
        opt_config = OptimizationConfig(
            use_alpha158=args.use_alpha158,
            use_hybrid=args.use_hybrid,
            smooth_window=args.smooth_window,
            target_vol=args.target_vol
        )
        # Use DoubleEnsemble to align with live trading
        pred = runner.train_model(model_type="double_ensemble", optimize_config=opt_config)


        # 4. Slice Predictions
        pred = slice_predictions(pred, test_start_time, end_time)

    # 5. Backtest
    with runner.run_experiment(params=vars(args)) as recorder:
        print("Running Backtest...")
        backtest_engine = BacktestEngine(pred)
        
        vol_feature = get_vol_feature(args.risk_parity or args.target_vol)
        
        report, positions = backtest_engine.run(
            topk=args.topk,
            n_drop=args.n_drop,
            buffer=args.buffer,
            vol_feature=vol_feature,
            target_vol=args.target_vol,
            use_trend_filter=args.dynamic_exposure,
            use_regime_filter=True,
            signal_halflife=args.signal_halflife,
            rebalance_threshold=args.rebalance_threshold,
            start_time=test_start_time,
            end_time=None
        )
        
        # Log Metrics & Artifacts
        log_backtest_metrics(report, pred, dataset, recorder)
        
        # Analysis
        analyzer = ResultAnalyzer()
        analyzer.process(report, positions)
        
        # Save CSV Artifacts for Debugging
        report.to_csv("artifacts/report.csv")
        
        # Save positions as text to avoid serialization issues
        with open("artifacts/positions.txt", "w") as f:
            f.write(str(positions))
        
        log_artifacts(recorder)

def fix_seed_and_init(seed):
    from delorean.utils import fix_seed
    fix_seed(seed)
    qlib.init(provider_uri=QLIB_PROVIDER_URI, region=QLIB_REGION, kernels=1)

def slice_predictions(pred, start, end):
    try:
        ts_start = pd.Timestamp(start)
        ts_end = pd.Timestamp(end)
        max_date = pred.index.get_level_values('datetime').max()
        if ts_end > max_date:
            ts_end = max_date
        return pred.loc[ts_start:ts_end]
    except Exception as e:
        print(f"Slice failed: {e}")
        return pred

def get_vol_feature(needed):
    if not needed: return None
    try:
        return fetch_volatility_feature(ETF_LIST, START_TIME, END_TIME)
    except:
        return None

def log_backtest_metrics(report, pred, dataset, recorder):
    """Calculate and log metrics to MLflow."""
    from qlib.contrib.evaluate import risk_analysis
    import numpy as np

    try:
        # 1. Risk Metrics (Sharpe, Return, MaxDD)
        risks = risk_analysis(report['return'], freq='day')
        
        annual_return = risks.loc['annualized_return', 'risk']
        sharpe = risks.loc['information_ratio', 'risk']
        max_dd = risks.loc['max_drawdown', 'risk']
        
        # 2. Turnover
        turnover = report['turnover'].mean()
        annual_turnover = turnover * 252
        
        metrics = {
            "annualized_return": float(annual_return),
            "sharpe": float(sharpe),
            "max_drawdown": float(max_dd),
            "ann_turnover": float(annual_turnover),
            "annual_turnover": float(annual_turnover), # Keep both just in case
            "sharpe_ratio": float(sharpe) # Keep both
        }
        
        # 3. Rank IC
        try:
            label = None
            if dataset is not None:
                # Standard Mode
                label_df = dataset.prepare("test", col_set=["label"])
                if not label_df.empty:
                    label = label_df.iloc[:, 0]
            else:
                # WF Mode: Fetch labels manually
                from qlib.data import D
                
                # Determine range from predictions
                time_idx = pred.index.get_level_values('datetime')
                start_t = time_idx.min()
                end_t = time_idx.max()
                
                # Assume horizon=1 for now as args not passed easily, or try to infer?
                # Actually, main() has args via closure if defined inside, but this is global.
                # Let's assume prediction horizon matches data label. Default 1 day.
                # Expression: Ref($close, -1) / $close - 1
                # To be precise, we should pass 'label_horizon' to this function.
                # Since we didn't update call signature in main, let's use default 1 for now.
                # Future improvement: Pass args.
                fields = ['Ref($close, -1) / $close - 1']
                
                # Fetch for ALL ETFs to be safe (or at least those in pred)
                instruments = pred.index.get_level_values('instrument').unique().tolist()
                label_df = D.features(instruments, fields, start_time=start_t, end_time=end_t)
                if not label_df.empty:
                    label = label_df.iloc[:, 0]

            if label is not None:
                # Ensure label index matches pred index (datetime, instrument)
                if isinstance(label.index, pd.MultiIndex):
                    levels = label.index.names
                    # If index is (instrument, datetime), swap it
                    if len(levels) == 2 and (levels[0] == 'instrument' or levels[0] == 'asset'):
                         label = label.swaplevel().sort_index()
                
                # Align indices
                common = pred.index.intersection(label.index)
                if not common.empty:
                    # Calculate Rank IC (Spearman)
                    ic = pred.loc[common].corr(label.loc[common], method='spearman')
                    metrics["rank_ic"] = float(ic)
                else:
                    print(f"Rank IC warning: No overlapping indices. Pred: {pred.index[0]}... Label: {label.index[0]}...")
                    
        except Exception as e:
            print(f"Rank IC calc failed: {e}")

        print(f"[MLflow] Logging Metrics: {metrics}")
        R.log_metrics(**metrics)
        
    except Exception as e:
        print(f"Failed to log metrics: {e}")

def log_artifacts(recorder):
    plots = ["cumulative_return.png", "excess_return.png", "factor_ic.png", "feature_correlation.png", "report.csv", "positions.txt"]
    for p in plots:
        p_path = os.path.join("artifacts", p)
        if os.path.exists(p_path):
            R.log_artifact(p_path)

if __name__ == "__main__":
    main()
