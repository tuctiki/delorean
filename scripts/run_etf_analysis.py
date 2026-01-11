import argparse
import qlib
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from delorean.config import QLIB_PROVIDER_URI, QLIB_REGION, BENCHMARK, START_TIME, END_TIME
from delorean.data import ETFDataLoader
from delorean.model import ModelTrainer
from delorean.backtest import BacktestEngine
from delorean.analysis import ResultAnalyzer, FactorAnalyzer
from delorean.experiment_manager import ExperimentManager
from delorean.feature_selection import FeatureSelector
from qlib.workflow import R

def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Run ETF Strategy Analysis")
    parser.add_argument("--topk", type=int, default=4, help="Number of stocks to hold in TopK strategy (default: 4)")
    parser.add_argument("--use_alpha158", action="store_true", help="Use Qlib Alpha158 embedded factors")
    parser.add_argument("--use_hybrid", action="store_true", help="Use Hybrid Factors (Custom + Alpha158)")
    return parser.parse_args()

def main() -> None:
    """
    Main execution function.
    """
    args = parse_args()

    # 1. Initialize Qlib
    qlib.init(provider_uri=QLIB_PROVIDER_URI, region=QLIB_REGION)

    # 2. Data Loading
    data_loader = ETFDataLoader(use_alpha158=args.use_alpha158, use_hybrid=args.use_hybrid)
    dataset = data_loader.load_data()

    # 3. Factor Analysis
    factor_analyzer = FactorAnalyzer()
    factor_analyzer.analyze(dataset)

    # 4. Model Training
    model_trainer = ModelTrainer()
    
    # Using Recorder for experiment tracking
    # Using Recorder for experiment tracking
    if args.use_hybrid:
        exp_name = "ETF_Strategy_Hybrid"
    elif args.use_alpha158:
        exp_name = "ETF_Strategy_Alpha158"
    else:
        exp_name = "ETF_Strategy_Refactored"
        
    with R.start(experiment_name=exp_name) as recorder:
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

        # Signal Smoothing
        print("Applying 10-day EWMA Signal Smoothing...")
        # Check index names (usually datetime, instrument)
        if pred.index.names[1] == 'instrument':
            level_name = 'instrument'
        else:
            level_name = pred.index.names[1]
            
        # EWMA Smoothing (Halflife=10 days)
        # Note: groupby(level=...).apply(...) often prepends the group key to the index.
        # Original index: (datetime, instrument)
        # Result index: (instrument, datetime, instrument) -> we need to drop top level.
        # 2. Time-Series Smoothing (EWMA on Raw Scores)
        # Reverting to Raw Score + 10d EWMA (Apex Config).
        # Rank smoothing removed to preserve magnitude signal.
        pred = pred.groupby(level=level_name).apply(
            lambda x: x.ewm(halflife=10, min_periods=1).mean()
        )
        
        # Remove redundant level 0 if added
        if pred.index.nlevels > 2:
            pred = pred.droplevel(0)
            
        # Ensure (datetime, instrument) order
        if pred.index.names[0] != 'datetime' and 'datetime' in pred.index.names:
             pred = pred.swaplevel()
             
        # Safe Sort by datetime
        pred = pred.dropna().sort_index()

        # --- Market Regime Filter (MA200) ---
        print("Calculating Market Regime Signal (HS300 MA200)...")
        from qlib.data import D
        from delorean.config import BENCHMARK, START_TIME, END_TIME
        
        # Load Benchmark Close Price
        # D.features returns MultiIndex (instrument, datetime)
        bench_df = D.features([BENCHMARK], ['$close'], start_time=START_TIME, end_time=END_TIME)
        
        # Compute MA200
        # Reset index to get 'datetime' as column or index level for easier handling
        bench_close = bench_df.droplevel(0) # Drop instrument level, now index is datetime
        bench_close.columns = ['close']
        
        # Calculate Moving Average (MA60 for faster response)
        bench_close['ma60'] = bench_close['close'].rolling(window=60).mean()
        
        # Define Regime: True = Bull (Close > MA60), False = Bear (Close <= MA60)
        market_regime = bench_close['close'] > bench_close['ma60']
        
        # 4. Backtest
        backtest_engine = BacktestEngine(pred)
        
        # Strategy Parameters
        strategy_params = {
            "topk": args.topk,
            "drop_rate": 0.96,
            "n_drop": 1,
            "market_regime": "MA60" # Descriptive for logging
        }
        
        # Pass updated robust params AND Market Regime (User Requested)
        report, positions = backtest_engine.run(
            topk=strategy_params["topk"], 
            drop_rate=strategy_params["drop_rate"], 
            n_drop=strategy_params["n_drop"], 
            market_regime=market_regime
        )
        
        # 5. Experiment Logging
        exp_manager = ExperimentManager()
        exp_manager.log_config(
            args=args,
            model_params=model_trainer.get_params(),
            strategy_params=strategy_params
        )
        exp_manager.save_report(report, positions)

        # 6. Analysis

        # 6. Analysis
        analyzer = ResultAnalyzer()
        analyzer.process(report, positions)

if __name__ == "__main__":
    main()
