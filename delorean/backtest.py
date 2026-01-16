from typing import List, Tuple, Dict, Any, Optional
from qlib.contrib.strategy.signal_strategy import BaseSignalStrategy
from qlib.backtest import backtest as qlib_backtest
from qlib.backtest import executor as qlib_executor
from qlib.backtest.decision import Order, OrderDir, TradeDecisionWO
from .config import TEST_START_TIME, END_TIME, BENCHMARK, ETF_LIST
from .strategy.portfolio import PortfolioOptimizer
from .strategy.execution import ExecutionModel
import pandas as pd
import copy
import random
import logging

logger = logging.getLogger(__name__)

class SimpleTopkStrategy(BaseSignalStrategy):
    """
    Robust TopK strategy with deterministic turnover control using buffer and n_drop.
    
    Attributes:
        topk (int): Number of stocks to hold.
        risk_degree (float): Percentage of capital to invest.
        n_drop (int): Maximum number of stocks to swap per trading step if trading occurs.
    """
    def __init__(self, topk: int = 4, risk_degree: float = 0.95, n_drop: int = 1, buffer: int = 2, 
                 vol_feature: pd.Series = None, trend_feature: pd.Series = None, regime_feature: pd.Series = None, 
                 target_vol: float = None, **kwargs: Any):
        super().__init__(risk_degree=risk_degree, **kwargs)
        self.topk = topk
        self.n_drop = n_drop
        self.buffer = buffer
        self.vol_feature = vol_feature 
        self.trend_feature = trend_feature # Per-Asset Trend
        self.regime_feature = regime_feature # Market Regime (Macro)
        self.target_vol = target_vol 
        
        # Initialize Sub-Models
        self.optimizer = PortfolioOptimizer(topk=topk, risk_degree=risk_degree)
        self.execution = ExecutionModel(topk=topk, buffer=buffer, n_drop=n_drop)

    def generate_trade_decision(self, execute_result: Any = None) -> TradeDecisionWO:
        """
        Generate trade orders with turnover control.
        """
        # 1. Get Trading Step and Time
        trade_step = self.trade_calendar.get_trade_step()
        trade_start_time, trade_end_time = self.trade_calendar.get_step_time(trade_step)
        
        # 2. Get Scores 
        # (Drop logic removed as unnecessary)

        
        # 2. Get Scores
        pred_score = self._get_pred_scores(trade_step)
        if pred_score is None:
            return TradeDecisionWO([], self)
            
        # 3. Filter Candidates (Trend Filter)
        # This logic could belong to a signal processor/AlphaModel
        # 3. Filter Candidates
        
        # [NEW] Market Regime Filter (Macro)
        # Now used for Asymmetric Volatility Scaling (not binary cutoff)
        current_regime = None  # Initialize for later use in calculate_weights
        if self.regime_feature is not None:
             try:
                 # Check regime for TODAY (trade_start_time)
                 # regime_feature is a Series of (date -> float ratio)
                 if trade_start_time in self.regime_feature.index:
                     current_regime = self.regime_feature.loc[trade_start_time]
                     # Note: No longer "Force Liquidate" here. 
                     # Asymmetric Vol Scaling in PortfolioOptimizer handles bear exposure.
             except Exception as e:
                 logger.debug(f"Regime filter failed for {trade_start_time}: {e}")

        # [EXISTING] Trend Filter (Micro / Per-Asset)
        if self.trend_feature is not None:
            try:
                current_trends = self.trend_feature.loc[trade_start_time]
                # Identify valid (Bull) stocks: Ratio > 1.0
                # Align logic with index handling
                if isinstance(pred_score, pd.Series):
                    # Align indices
                    common = pred_score.index.intersection(current_trends.index)
                    if not common.empty:
                         trends_aligned = current_trends.loc[common]
                         valid_mask = trends_aligned > 1.0
                         pred_score = pred_score.loc[common[valid_mask]]         
            except Exception as e:
                logger.debug(f"Trend filter failed for {trade_start_time}: {e}")

        # 4. Portfolio Optimization (Target Weights)
        # Sort and take top K candidates for weighting
        sorted_score = pred_score.sort_values(ascending=False)
        target_stocks = sorted_score.head(self.topk).index.tolist()
        
        target_weights = self.optimizer.calculate_weights(
            target_stocks, 
            trade_start_time, 
            vol_feature=self.vol_feature,
            target_vol=self.target_vol,
            regime_ratio=current_regime if self.regime_feature is not None else None
        )
        
        # 5. Execution (Generate Orders)
        orders = self.execution.generate_orders(
            current_pos=self.trade_position,
            pred_score=pred_score,
            target_weights=target_weights,
            trade_exchange=self.trade_exchange,
            trade_start_time=trade_start_time,
            trade_end_time=trade_end_time
        )

        return TradeDecisionWO(orders, self)

    def _get_pred_scores(self, trade_step: int) -> Optional[pd.Series]:
        """Fetch prediction scores for the next trading day (shift=1)."""
        pred_start_time, pred_end_time = self.trade_calendar.get_step_time(trade_step, shift=1)
        pred_score = self.signal.get_signal(start_time=pred_start_time, end_time=pred_end_time)
        
        if isinstance(pred_score, pd.DataFrame):
            pred_score = pred_score.iloc[:, 0]
            
        return pred_score

    def _get_target_stocks(self, pred_score: pd.Series) -> pd.Index:
        """Identify Top K stocks."""
        return pred_score.sort_values(ascending=False).head(self.topk).index





class BacktestEngine:
    """
    Orchestrates the backtesting process using Qlib's standard backtest engine.
    """
    def __init__(self, pred: pd.Series):
        """
        Args:
            pred (pd.Series): Prediction scores indexed by (datetime, instrument).
        """
        self.pred = pred

    def run(self, topk: int = 3, start_time=None, end_time=None, use_trend_filter: bool = False, use_regime_filter: bool = True, **kwargs: Any) -> Tuple[pd.DataFrame, Dict[Any, Any]]:
        """
        Run the backtest simulation.

        Args:
            topk (int): Number of stocks to hold in the TopK strategy.
            start_time (str|pd.Timestamp): Custom start time for backtest. Defaults to config.TEST_START_TIME.
            end_time (str|pd.Timestamp): Custom end time for backtest. Defaults to derived from data.
            end_time (str|pd.Timestamp): Custom end time for backtest. Defaults to derived from data.
            use_trend_filter (bool): Whether to enable the per-asset trend filter (Default: False).
            use_regime_filter (bool): Whether to enable the market regime trend filter (Default: False).
            target_vol (float): Annualized Target Volatility (e.g. 0.20 for 20%). None = No targeting.
            **kwargs: Additional strategy parameters (drop_rate, n_drop).

        Returns:
            Tuple[pd.DataFrame, dict]: A tuple containing the backtest report DataFrame 
            and a dictionary of position details.
        """

        print(f"\nRunning Backtest (Custom SimpleTopkStrategy, TopK={topk}, Trend={'On' if use_trend_filter else 'Off'}, Regime={'On' if use_regime_filter else 'Off'}, Params={kwargs})...")
        # Fetch Trend Feature if needed
        trend_feature = None
        if use_trend_filter:
            # We always try to fetch it for per-asset filtering
            from qlib.data import D
            # Price/MA60 Ratio
            # We need to fetch it aligned with pred's range
            # To be safe, fetch for the whole range of pred
            
            # Extract time range from pred index
            time_idx = self.pred.index.get_level_values('datetime')
            start_t = time_idx.min()
            end_t = time_idx.max()
            
            try:
                # Fetch Price_MA60_Ratio
                # Expression: $close / Mean($close, 60)
                fields = ['$close / Mean($close, 60)']
                names = ['ma60_ratio']
                
                # We fetch for ALL ETFs in the list, to cover whatever is in pred
                # Actually pred might contain subset if selection used
                # But querying ETF_LIST is safer
                feat_df = D.features(ETF_LIST, fields, start_time=start_t, end_time=end_t)
                if not feat_df.empty:
                    # Index is (datetime, instrument)
                    trend_feature = feat_df.iloc[:, 0] # Series
            except Exception as e:
                print(f"Warning: Failed to fetch Trend Feature/Data for backtest: {e}")

        # Fetch Regime Feature if needed
        regime_feature = None
        if use_regime_filter:
            from qlib.data import D
            # Expression: $close / Mean($close, 60) for BENCHMARK
            try:
                # Need consistent time range
                time_idx = self.pred.index.get_level_values('datetime')
                start_t = time_idx.min()
                end_t = time_idx.max()
                
                fields = ['$close / Mean($close, 60)']
                # Fetch for Benchmark Only
                feat_df = D.features([BENCHMARK], fields, start_time=start_t, end_time=end_t)
                if not feat_df.empty:
                     # Index is (datetime, instrument). We want Series indexed by datetime
                     # Since it's one instrument, we can just reset index or xs
                     regime_feature = feat_df.xs(BENCHMARK, level='instrument')['$close / Mean($close, 60)']
            except Exception as e:
                 print(f"Warning: Failed to fetch Regime Feature for backtest: {e}")


        # Strategy Config
        STRATEGY_CONFIG = {
            "topk": topk,
            "risk_degree": 0.95,
            "signal": self.pred,
            "trend_feature": trend_feature,
            "regime_feature": regime_feature,
            "target_vol": kwargs.get('target_vol', None),
            **kwargs # Pass drop_rate and n_drop
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
        strategy_obj = SimpleTopkStrategy(**STRATEGY_CONFIG)
        executor_obj = qlib_executor.SimulatorExecutor(**EXECUTOR_CONFIG["kwargs"])

        # Determine Backtest Time Range
        if start_time is None:
            start_time = TEST_START_TIME
        
        if end_time is None:
            # The backtest engine's calendar logic requires the 'next' step to exist to define the interval.
            valid_dates = self.pred.index.get_level_values('datetime').unique().sort_values()
            if len(valid_dates) > 1:
                end_time = valid_dates[-2]
            else:
                end_time = valid_dates[-1]

        portfolio_metric_dict, indicator_dict = qlib_backtest(
            executor=executor_obj,
            strategy=strategy_obj,
            start_time=start_time,
            end_time=end_time,
            account=1000000,
            benchmark=BENCHMARK,
        )
        
        # Extract report and positions (key is usually '1day' for daily frequency)
        # portfolio_metric_dict mapping: freq -> (report_df, positions_dict)
        report, positions = list(portfolio_metric_dict.values())[0]
        
        return report, positions
