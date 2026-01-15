import argparse
import pandas as pd
import qlib
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from delorean.config import QLIB_PROVIDER_URI, QLIB_REGION, BENCHMARK, START_TIME, END_TIME, DEFAULT_EXPERIMENT_NAME, TEST_START_TIME, TRAIN_END_TIME
from delorean.data import ETFDataLoader
from delorean.model import ModelTrainer
from delorean.backtest import BacktestEngine
from delorean.analysis import ResultAnalyzer, FactorAnalyzer
from delorean.experiment_manager import ExperimentManager
from delorean.feature_selection import FeatureSelector
from delorean.signals import smooth_predictions
from delorean.utils import fix_seed
from qlib.workflow import R

def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Run ETF Strategy Analysis")
    parser.add_argument("--topk", type=int, default=4, help="Number of stocks to hold in TopK strategy (default: 4)")
    parser.add_argument("--use_alpha158", action="store_true", help="Use Qlib Alpha158 embedded factors")
    parser.add_argument("--use_hybrid", action="store_true", help="Use Hybrid Factors (Custom + Alpha158)")
    parser.add_argument("--risk_parity", action="store_true", help="Enable Volatility Targeting (1/Vol)")
    parser.add_argument("--dynamic_exposure", action="store_true", help="Enable Trend-based Dynamic Exposure")
    parser.add_argument("--buffer", type=int, default=2, help="Rank Buffer for Hysteresis (default: 2)")
    parser.add_argument("--label_horizon", type=int, default=1, help="Forward return label horizon in days (default: 1)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--smooth_window", type=int, default=10, help="EWMA smoothing halflife (days). Higher = Lower Turnover.")
    
    # Time Range Overrides
    parser.add_argument("--start_time", type=str, default=None, help="Backtest Data Start Time (e.g. 2015-01-01)")
    parser.add_argument("--train_end_time", type=str, default=None, help="Training End Time (e.g. 2021-12-31)")
    parser.add_argument("--test_start_time", type=str, default=None, help="Test Start Time (e.g. 2022-01-01)")
    parser.add_argument("--end_time", type=str, default=None, help="Backtest End Time (e.g. 2025-12-31)")
    parser.add_argument("--experiment_name", type=str, default=None, help="Custom MLflow Experiment Name")
    
    return parser.parse_args()

# fix_seed is now imported from delorean.utils

def main() -> None:
    """
    Main execution function.
    """
    args = parse_args()
    fix_seed(args.seed)


    # 1. Initialize Qlib
    qlib.init(provider_uri=QLIB_PROVIDER_URI, region=QLIB_REGION, kernels=1)

    # Time Config Overrides
    start_time = args.start_time if args.start_time else START_TIME
    end_time = args.end_time if args.end_time else END_TIME

    # [FIX] Clean up existing artifacts to prevent logging stale plots
    print("Cleaning up old artifacts...")
    plots = ["cumulative_return.png", "excess_return.png", "factor_ic.png", "feature_correlation.png"]
    for p in plots:
        p_path = os.path.join("artifacts", p)
        if os.path.exists(p_path):
            os.remove(p_path)


    # 2. Data Loading
    data_loader = ETFDataLoader(
        use_alpha158=args.use_alpha158, 
        use_hybrid=args.use_hybrid,
        label_horizon=args.label_horizon,
        start_time=start_time,
        end_time=end_time
    )
    
    dataset = data_loader.load_data(
        train_start=start_time,
        train_end=args.train_end_time,
        test_start=args.test_start_time,
        test_end=end_time
    )

    # 3. Factor Analysis
    factor_analyzer = FactorAnalyzer()
    factor_analyzer.analyze(dataset)

    # 4. Model Training
    model_trainer = ModelTrainer(seed=args.seed)
    
    # Using Recorder for experiment tracking
    # Always use the default experiment name for consistent tracking
    exp_name = DEFAULT_EXPERIMENT_NAME
        
    with R.start(experiment_name=exp_name) as recorder:
        # Log all arguments as experiment parameters
        # Ensure all values are strings for MLflow compatibility
        params = {k: str(v) for k, v in vars(args).items()}
        print(f"Logging Params: {params}")
        R.log_params(**params)
        
        model_trainer.train(dataset)
        pred = model_trainer.predict(dataset)
        
        # Feature Importance
        feature_imp = model_trainer.get_feature_importance(dataset)
        print("\nTop 10 Feature Importance (Stage 1):\n", feature_imp.head(10))

        if args.use_alpha158 or args.use_hybrid:
            print("\n[Optimization] Performing Feature Selection (Top 20) with Correlation Filtering...")
            
            # 1. Start with Top 30 important features (to have buffer for filtering)
            initial_top_k = 40 
            top_features_initial = feature_imp['feature'].head(initial_top_k).tolist()
            print(f"Initial Top {initial_top_k} Candidates: {top_features_initial}")
            
            # 2. Filter using centralized FeatureSelector
            top_features = FeatureSelector.filter_by_correlation(dataset, top_features_initial, threshold=0.95)
            
            # Ensure we only keep Top 20 from the filtered list (FeatureSelector returns all non-correlated)
            top_features = top_features[:20]
            print(f"Final Selected Features ({len(top_features)}): {top_features}")
            
            print("Stage 2: Retraining with Selected Features...")
            model_trainer.train(dataset, selected_features=top_features)
            
            # Update predictions with new model
            pred = model_trainer.predict(dataset)
            
            # Re-check importance (Optional)
            # feature_imp_opt = model_trainer.get_feature_importance(dataset)
            # print("\nTop 10 Feature Importance (Stage 2):\n", feature_imp_opt.head(10))

        # Signal Smoothing using shared module
        print(f"Applying {args.smooth_window}-day EWMA Signal Smoothing...")
        pred = smooth_predictions(pred, halflife=args.smooth_window)
        
        # [FIX] Enforce Test Range Slicing
        # Ensure backtest only runs on the requested test period
        test_start = args.test_start_time if args.test_start_time else TEST_START_TIME
        test_end = args.end_time if args.end_time else END_TIME
        
        print(f"Enforcing Backtest Range: {test_start} to {test_end}")
        
        # Pred index is (datetime, instrument), sorted. Slicing logic:
        try:
            # Flexible date parsing
            ts_start = pd.Timestamp(test_start)
            ts_end = pd.Timestamp(test_end)
            
            # Cap end time at data availability to avoid IndexErrors in Qlib
            max_data_date = pred.index.get_level_values('datetime').max()
            if ts_end > max_data_date:
                print(f"Requested end time {ts_end} exceeds data availability ({max_data_date}). Capping to max data date.")
                ts_end = max_data_date
                # Update test_end string for consistency if passed to run()
                test_end = ts_end.strftime('%Y-%m-%d')
                
            pred = pred.loc[ts_start:ts_end]
            print(f"Predictions sliced. New shape: {pred.shape}")
        except Exception as e:
            print(f"Warning: Failed to slice predictions by date: {e}")

        # 4. Backtest
        backtest_engine = BacktestEngine(pred)
        
        # Strategy Parameters
        strategy_params = {
            "topk": args.topk,
            "drop_rate": 0.96,
            "n_drop": 1,
            "buffer": args.buffer,
            "risk_parity": args.risk_parity,
            "dynamic_exposure": args.dynamic_exposure
        }
        
        # Prepare Position Control Data
        vol_feature = None
        if args.risk_parity:
            # We need VOL20. 
            print("Fetching Volatility Data for Risk Parity...")
            # Fetch explicitly via Qlib
            try:
                from qlib.data import D
                # Std($close/Ref($close,1)-1, 20)
                # Note: Qlib expressions need to be exact.
                vol_data = D.features(D.instruments(market=QLIB_REGION), ['Std($close/Ref($close,1)-1, 20)'], start_time=START_TIME, end_time=END_TIME)
                vol_data.columns = ['VOL20']
                vol_feature = vol_data['VOL20']
            except Exception as e:
                print(f"Warning: Failed to fetch VOL20 data: {e}")
                vol_feature = None

        # Run Backtest
        report, positions = backtest_engine.run(
            topk=strategy_params["topk"], 
            drop_rate=strategy_params["drop_rate"], 
            n_drop=strategy_params["n_drop"], 
            buffer=strategy_params["buffer"],
            vol_feature=vol_feature,
            start_time=test_start,
            end_time=None # Let Engine determine safe end date from sliced pred
        )
        
        # Log key backtest metrics to MLflow
        try:
            from qlib.contrib.evaluate import risk_analysis
            # Ensure report is a DataFrame and has 'return'
            if isinstance(report, pd.DataFrame) and 'return' in report.columns:
                risk_df = risk_analysis(report['return'])
                sharpe = risk_df.loc['sharpe', 'risk'] if 'sharpe' in risk_df.index else None
                if sharpe is None:
                     # Fallback for different Qlib versions
                     sharpe = risk_df.loc['information_ratio', 'risk'] if 'information_ratio' in risk_df.index else None
                
                # Calculate Rank IC from predictions
                from delorean.utils import calculate_rank_ic
                
                # Use 'test' segment labels
                labels = dataset.prepare("test", col_set="label", data_key="infer")
                rank_ic = calculate_rank_ic(pred, labels)
                
                if sharpe is not None:
                    R.log_metrics(sharpe=float(sharpe))
                if rank_ic is not None:
                    R.log_metrics(rank_ic=float(rank_ic))
                
                # Log turnover metrics
                if 'turnover' in report.columns:
                    avg_turnover = report['turnover'].mean()
                    ann_turnover = avg_turnover * 252
                    trading_days = int((report['turnover'] > 0).sum())
                    R.log_metrics(
                        daily_turnover=float(avg_turnover),
                        ann_turnover=float(ann_turnover),
                        trading_days=float(trading_days)
                    )
                    print(f"Logged metrics to MLflow: sharpe={sharpe}, rank_ic={rank_ic}, ann_turnover={ann_turnover:.2%}")
                else:
                    print(f"Logged metrics to MLflow: sharpe={sharpe}, rank_ic={rank_ic}")
            else:
                print("Warning: Report format invalid for metric logging")
        except Exception as e:
            print(f"Warning: Could not log metrics to MLflow: {e}")
        
        # 5. Experiment Logging
        exp_manager = ExperimentManager()
        exp_manager.log_config(
            args=args,
            model_params=model_trainer.get_params(),
            strategy_params=strategy_params
        )
        exp_manager.save_report(report, positions)
        
        # 6. Analysis (Generate Plots FIRST)
        analyzer = ResultAnalyzer()
        analyzer.process(report, positions)
        
        # Log Plots as Artifacts (explicitly)
        # Note: These are saved by ResultAnalyzer into OUTPUT_DIR (artifacts/)
        # We need to log them to the recorder for MLflow tracking
        plots = ["cumulative_return.png", "excess_return.png", "factor_ic.png", "feature_correlation.png"]
        for p in plots:
             p_path = os.path.join("artifacts", p) # Analyzer saves to artifacts/
             if os.path.exists(p_path):
                 R.log_artifact(p_path)

        # 6. Analysis


if __name__ == "__main__":
    main()
