from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy
from qlib.backtest import backtest as qlib_backtest
from qlib.backtest import executor as qlib_executor
from constants import TEST_START_TIME, END_TIME, BENCHMARK
import pandas as pd

class BacktestEngine:
    def __init__(self, pred):
        self.pred = pred

    def run(self):
        print("\nRunning Backtest (Optimized TopkDropoutStrategy)...")
        # Strategy Config
        STRATEGY_CONFIG = {
            "topk": 3,
            "n_drop": 1,
            "risk_degree": 0.95,
            "hold_thresh": 5,        # Hold at least 5 days to reduce turnover
            "signal": self.pred # Use stored predictions
        }

        EXECUTOR_CONFIG = {
            "class": "SimulatorExecutor",
            "module_path": "qlib.backtest.executor",
            "kwargs": {
                "time_per_step": "day",
                "generate_portfolio_metrics": True,
                "verbose": True,
                "track_data": True,
                "limit_threshold": 0.095,
                "deal_price": "close",
                "open_cost": 0.0003,
                "close_cost": 0.0003,
                "min_cost": 0,
            },
        }

        # Backtest Execution
        strategy_obj = TopkDropoutStrategy(**STRATEGY_CONFIG)
        executor_obj = qlib_executor.SimulatorExecutor(**EXECUTOR_CONFIG["kwargs"])

        portfolio_metric_dict, indicator_dict = qlib_backtest(
            executor=executor_obj,
            strategy=strategy_obj,
            start_time=TEST_START_TIME,
            end_time=END_TIME,
            account=1000000,
            benchmark=BENCHMARK,
        )
        
        # Extract report and positions (key is usually '1day' for daily frequency)
        # portfolio_metric_dict mapping: freq -> (report_df, positions_dict)
        report, positions = list(portfolio_metric_dict.values())[0]
        
        return report, positions
