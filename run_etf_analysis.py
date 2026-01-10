import argparse
import qlib
from constants import QLIB_PROVIDER_URI, QLIB_REGION
from data import ETFDataLoader
from model import ModelTrainer
from backtest import BacktestEngine
from analysis import ResultAnalyzer, FactorAnalyzer
from qlib.workflow import R

def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Run ETF Strategy Analysis")
    parser.add_argument("--topk", type=int, default=3, help="Number of stocks to hold in TopK strategy (default: 3)")
    return parser.parse_args()

def main() -> None:
    """
    Main execution function.
    """
    args = parse_args()

    # 1. Initialize Qlib
    qlib.init(provider_uri=QLIB_PROVIDER_URI, region=QLIB_REGION)

    # 2. Data Loading
    data_loader = ETFDataLoader()
    dataset = data_loader.load_data()

    # 3. Factor Analysis
    factor_analyzer = FactorAnalyzer()
    factor_analyzer.analyze(dataset)

    # 4. Model Training
    model_trainer = ModelTrainer()
    
    # Using Recorder for experiment tracking
    with R.start(experiment_name="ETF_Strategy_Refactored") as recorder:
        model_trainer.train(dataset)
        pred = model_trainer.predict(dataset)
        
        # Feature Importance
        feature_imp = model_trainer.get_feature_importance(dataset)
        print("\nTop 10 Feature Importance:\n", feature_imp.head(10))

        # Signal Smoothing (Experiment 4 + Aggressive 10-day EWMA)
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

        # 4. Backtest
        backtest_engine = BacktestEngine(pred)
        # Pass updated robust params (TopK=4, DropRate=0.96, NDrop=1)
        report, positions = backtest_engine.run(topk=4, drop_rate=0.96, n_drop=1)

        # 5. Analysis
        analyzer = ResultAnalyzer()
        analyzer.process(report, positions)

if __name__ == "__main__":
    main()
