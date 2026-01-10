import qlib
from constants import QLIB_PROVIDER_URI, QLIB_REGION
from data import ETFDataLoader
from model import ModelTrainer
from backtest import BacktestEngine
from analysis import ResultAnalyzer
from qlib.workflow import R

if __name__ == "__main__":
    # 1. Initialize Qlib
    qlib.init(provider_uri=QLIB_PROVIDER_URI, region=QLIB_REGION)

    # 2. Data Loading
    data_loader = ETFDataLoader()
    dataset = data_loader.load_data()

    # 3. Model Training
    model_trainer = ModelTrainer()
    
    # Using Recorder for experiment tracking
    with R.start(experiment_name="ETF_Strategy_Refactored") as recorder:
        model_trainer.train(dataset)
        pred = model_trainer.predict(dataset)
        
        # Feature Importance
        feature_imp = model_trainer.get_feature_importance(dataset)
        print("\nTop 10 Feature Importance:\n", feature_imp.head(10))

        # 4. Backtest
        backtest_engine = BacktestEngine(pred)
        report, positions = backtest_engine.run()

        # 5. Analysis
        analyzer = ResultAnalyzer()
        analyzer.process(report, positions)
