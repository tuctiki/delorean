from typing import List, Tuple, Dict, Any, Optional
from qlib.contrib.strategy.signal_strategy import BaseSignalStrategy
from qlib.backtest import backtest as qlib_backtest
from qlib.backtest import executor as qlib_executor
from qlib.backtest.decision import Order, OrderDir, TradeDecisionWO
from constants import TEST_START_TIME, END_TIME, BENCHMARK
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

    def generate_trade_decision(self, execute_result: Any = None) -> TradeDecisionWO:
        """
        Generate trade orders for the current step.

        Args:
            execute_result: Previous execution result (unused in this strategy).

        Returns:
            TradeDecisionWO: A decision object containing a list of orders.
        """
        # 1. Get Trading Step and Time
        trade_step = self.trade_calendar.get_trade_step()
        trade_start_time, trade_end_time = self.trade_calendar.get_step_time(trade_step)
        
        # 2. Get Scores
        pred_score = self._get_pred_scores(trade_step)
        if pred_score is None:
            return TradeDecisionWO([], self)
            
        # 3. Determine Target Stocks (Top K)
        target_stocks = self._get_target_stocks(pred_score)
        
        # 4. Current Position Snapshot
        current_temp = copy.deepcopy(self.trade_position)
        
        # 5. Generate Orders
        sell_orders = self._generate_sell_orders(current_temp, target_stocks, trade_start_time, trade_end_time)
        
        # Update position after sell (simulated)
        for order in sell_orders:
            # Note: deal_order updates the position object in-place (current_temp)
            self.trade_exchange.deal_order(order, position=current_temp)
        
        buy_orders = self._generate_buy_orders(current_temp, target_stocks, trade_start_time, trade_end_time)
        
        return TradeDecisionWO(sell_orders + buy_orders, self)

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

    def _generate_sell_orders(self, current_pos: Any, target_stocks: pd.Index, 
                              start_time: pd.Timestamp, end_time: pd.Timestamp) -> List[Order]:
        """Generate sell orders for stocks not in the target list."""
        sell_order_list = []
        current_stock_list = current_pos.get_stock_list()
        
        for code in current_stock_list:
            if code not in target_stocks:
                if not self.trade_exchange.is_stock_tradable(stock_id=code, start_time=start_time, end_time=end_time, direction=OrderDir.SELL):
                    continue
                
                sell_amount = current_pos.get_stock_amount(code=code)
                sell_order = Order(
                    stock_id=code,
                    amount=sell_amount,
                    start_time=start_time,
                    end_time=end_time,
                    direction=Order.SELL
                )
                if self.trade_exchange.check_order(sell_order):
                    sell_order_list.append(sell_order)
        
        return sell_order_list

    def _generate_buy_orders(self, current_pos: Any, target_stocks: pd.Index, 
                             start_time: pd.Timestamp, end_time: pd.Timestamp) -> List[Order]:
        """Generate buy/adjust orders to reach equal weight for target stocks."""
        buy_order_list = []
        sell_order_rebalance_list = []
        
        # Calculate target value per stock (Equal Weight based on current total value)
        current_value = current_pos.calculate_value()
        target_value_per_stock = current_value * self.risk_degree / self.topk
        
        for code in target_stocks:
            if not self.trade_exchange.is_stock_tradable(stock_id=code, start_time=start_time, end_time=end_time, direction=OrderDir.BUY):
                continue
            
            # Current holding
            current_amount = current_pos.get_stock_amount(code=code)
            current_price = self.trade_exchange.get_deal_price(stock_id=code, start_time=start_time, end_time=end_time, direction=OrderDir.BUY)
            
            # Gap
            target_amount = target_value_per_stock / current_price
            msg_amount = target_amount - current_amount
            
            # Determine direction
            if msg_amount > 0:
                # Buy
                factor = self.trade_exchange.get_factor(stock_id=code, start_time=start_time, end_time=end_time)
                buy_amount = self.trade_exchange.round_amount_by_trade_unit(msg_amount, factor)
                if buy_amount > 0:
                    buy_order = Order(
                        stock_id=code,
                        amount=buy_amount,
                        start_time=start_time,
                        end_time=end_time,
                        direction=OrderDir.BUY
                    )
                    buy_order_list.append(buy_order)
            elif msg_amount < 0:
                 # Sell (Rebalance down)
                 sell_amount = abs(msg_amount)
                 sell_order = Order(
                    stock_id=code,
                    amount=sell_amount,
                    start_time=start_time,
                    end_time=end_time,
                    direction=Order.SELL
                 )
                 sell_order_rebalance_list.append(sell_order)

        # Combine rebalance sells and buys. 
        # Note: Strategy usually returns all orders.
        return sell_order_rebalance_list + buy_order_list

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

    def run(self, topk: int = 3) -> Tuple[pd.DataFrame, Dict[Any, Any]]:
        """
        Run the backtest simulation.

        Args:
            topk (int): Number of stocks to hold in the TopK strategy.

        Returns:
            Tuple[pd.DataFrame, dict]: A tuple containing the backtest report DataFrame 
            and a dictionary of position details.
        """
        print(f"\nRunning Backtest (Experiment 3: Custom SimpleTopkStrategy, TopK={topk})...")
        # Strategy Config
        STRATEGY_CONFIG = {
            "topk": topk,
            "risk_degree": 0.95,
            "signal": self.pred,
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
