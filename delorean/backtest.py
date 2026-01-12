from typing import List, Tuple, Dict, Any, Optional
from qlib.contrib.strategy.signal_strategy import BaseSignalStrategy
from qlib.backtest import backtest as qlib_backtest
from qlib.backtest import executor as qlib_executor
from qlib.backtest.decision import Order, OrderDir, TradeDecisionWO
from .config import TEST_START_TIME, END_TIME, BENCHMARK
from .strategy.portfolio import PortfolioOptimizer
from .strategy.execution import ExecutionModel
import pandas as pd
import copy

class SimpleTopkStrategy(BaseSignalStrategy):
    """
    A simplified TopK strategy that aims to reduce turnover.
    
    Logic:
    1. Selects Top K stocks based on prediction scores.
    2. Only swaps stocks if they fall out of the Top K (implicit in rebalancing).
    3. Rebalances strictly to target weights (Equal Weight) daily.
    
    Attributes:
        topk (int): Number of stocks to hold.
        risk_degree (float): Percentage of total capital to invest (e.g., 0.95).
    """
    def __init__(self, topk: int = 3, risk_degree: float = 0.95, **kwargs: Any):
        super().__init__(risk_degree=risk_degree, **kwargs)
        self.topk = topk

import random

class SimpleTopkStrategy(BaseSignalStrategy):
    """
    Robust TopK strategy with probabilistic retention to drastically reduce turnover.
    
    Attributes:
        topk (int): Number of stocks to hold.
        risk_degree (float): Percentage of capital to invest.
        drop_rate (float): Probability of **keeping** existing positions (skipping trading). 
                           User terminology: "dropout_rate" ~96% -> 96% chance keep old.
        n_drop (int): Maximum number of stocks to swap per trading step if trading occurs.
    """
    def __init__(self, topk: int = 4, risk_degree: float = 0.95, drop_rate: float = 0.96, n_drop: int = 1, buffer: int = 2, market_regime: pd.Series = None, vol_feature: pd.Series = None, trend_feature: pd.Series = None, **kwargs: Any):
        super().__init__(risk_degree=risk_degree, **kwargs)
        self.topk = topk
        self.drop_rate = drop_rate
        self.n_drop = n_drop
        self.buffer = buffer
        self.market_regime = market_regime
        self.vol_feature = vol_feature 
        self.trend_feature = trend_feature 
        
        # Initialize Sub-Models
        self.optimizer = PortfolioOptimizer(topk=topk, risk_degree=risk_degree)
        self.execution = ExecutionModel(topk=topk, buffer=buffer, n_drop=n_drop)

    def generate_trade_decision(self, execute_result: Any = None) -> TradeDecisionWO:
        """
        Generate trade orders with turnover control and market regime filter.
        """
        # 1. Get Trading Step and Time
        trade_step = self.trade_calendar.get_trade_step()
        trade_start_time, trade_end_time = self.trade_calendar.get_step_time(trade_step)

        # 0. Market Regime (Per-Asset Trend Filter Logic remains here or moves to AlphaModel?)
        # For now, keep high-level regime check here.
        if self.market_regime is not None:
             try:
                is_bull = self.market_regime.loc[trade_start_time] if trade_start_time in self.market_regime.index else True
             except:
                is_bull = True
             
             if not is_bull:
                 # Standard Regime Filter: Sell All (Cash)
                 return TradeDecisionWO(self._generate_sell_orders(self.trade_position, pd.Index([]), trade_start_time, trade_end_time), self)
        
        # 0.5 Turnover control: Probabilistic Retention (Pre-check)
        # Note: Ideally this should move to ExecutionModel too, but it short-circuits everything.
        current_risk_degree = self.risk_degree 
        current_holdings_list = self.trade_position.get_stock_list()
        should_force_trade = (len(current_holdings_list) == 0) and (current_risk_degree > 0)
        
        if (not should_force_trade) and (random.random() < self.drop_rate):
            return TradeDecisionWO([], self)
        
        # 2. Get Scores
        pred_score = self._get_pred_scores(trade_step)
        if pred_score is None:
            return TradeDecisionWO([], self)
            
        # 3. Filter Candidates (Trend Filter)
        # This logic could belong to a signal processor/AlphaModel
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
                pass 

        # 4. Portfolio Optimization (Target Weights)
        # Sort and take top K candidates for weighting
        sorted_score = pred_score.sort_values(ascending=False)
        target_stocks = sorted_score.head(self.topk).index.tolist()
        
        target_weights = self.optimizer.calculate_weights(
            target_stocks, 
            trade_start_time, 
            vol_feature=self.vol_feature
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

    def run(self, topk: int = 3, market_regime: pd.Series = None, start_time=None, end_time=None, use_trend_filter: bool = False, **kwargs: Any) -> Tuple[pd.DataFrame, Dict[Any, Any]]:
        """
        Run the backtest simulation.

        Args:
            topk (int): Number of stocks to hold in the TopK strategy.
            market_regime (pd.Series): Boolean series (True=Bull, False=Bear) to filter trades.
            start_time (str|pd.Timestamp): Custom start time for backtest. Defaults to config.TEST_START_TIME.
            end_time (str|pd.Timestamp): Custom end time for backtest. Defaults to derived from data.
            use_trend_filter (bool): Whether to enable the per-asset trend filter (Default: False).
            **kwargs: Additional strategy parameters (drop_rate, n_drop).

        Returns:
            Tuple[pd.DataFrame, dict]: A tuple containing the backtest report DataFrame 
            and a dictionary of position details.
        """

        print(f"\nRunning Backtest (Custom SimpleTopkStrategy, TopK={topk}, GlobalRegime={'On' if market_regime is not None else 'Off'}, PerAssetTrend={'On' if use_trend_filter else 'Off'}, Params={kwargs})...")
        # Fetch Trend Feature if needed
        trend_feature = None
        if use_trend_filter:
            # We always try to fetch it for per-asset filtering
            from qlib.data import D
            from .config import ETF_LIST
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

        # Strategy Config
        STRATEGY_CONFIG = {
            "topk": topk,
            "risk_degree": 0.95,
            "signal": self.pred,
            "market_regime": market_regime,
            "trend_feature": trend_feature,
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
