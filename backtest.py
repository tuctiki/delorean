from qlib.contrib.strategy.signal_strategy import BaseSignalStrategy
from qlib.backtest import backtest as qlib_backtest
from qlib.backtest import executor as qlib_executor
from qlib.backtest.decision import Order, OrderDir, TradeDecisionWO
from constants import TEST_START_TIME, END_TIME, BENCHMARK
import pandas as pd
import copy

class SimpleTopkStrategy(BaseSignalStrategy):
    def __init__(self, topk=3, risk_degree=0.95, **kwargs):
        super().__init__(risk_degree=risk_degree, **kwargs)
        self.topk = topk

    def generate_trade_decision(self, execute_result=None):
        # 1. Get Trading Step and Time
        trade_step = self.trade_calendar.get_trade_step()
        trade_start_time, trade_end_time = self.trade_calendar.get_step_time(trade_step)
        pred_start_time, pred_end_time = self.trade_calendar.get_step_time(trade_step, shift=1)
        
        # 2. Get Scores
        pred_score = self.signal.get_signal(start_time=pred_start_time, end_time=pred_end_time)
        if isinstance(pred_score, pd.DataFrame):
            pred_score = pred_score.iloc[:, 0]
        if pred_score is None:
            return TradeDecisionWO([], self)
            
        # 3. Determine Target Stocks (Top K)
        target_stocks = pred_score.sort_values(ascending=False).head(self.topk).index
        
        # 4. Current Position
        current_temp = copy.deepcopy(self.trade_position)
        current_stock_list = current_temp.get_stock_list()
        
        sell_order_list = []
        buy_order_list = []
        cash = current_temp.get_cash()
        
        # 5. Sell Logic: Sell anything not in target
        for code in current_stock_list:
            if code not in target_stocks:
                # check tradable
                if not self.trade_exchange.is_stock_tradable(stock_id=code, start_time=trade_start_time, end_time=trade_end_time, direction=OrderDir.SELL):
                    continue
                
                sell_amount = current_temp.get_stock_amount(code=code)
                sell_order = Order(
                    stock_id=code,
                    amount=sell_amount,
                    start_time=trade_start_time,
                    end_time=trade_end_time,
                    direction=Order.SELL
                )
                if self.trade_exchange.check_order(sell_order):
                    sell_order_list.append(sell_order)
                    trade_val, trade_cost, trade_price = self.trade_exchange.deal_order(sell_order, position=current_temp)
                    cash += trade_val - trade_cost
        
        # 6. Buy Logic: Buy target stocks to equal weight
        # Calculate target value per stock
        # We assume equal weight for simplicity for now
        # Total Value = Cash (after sell) + Value of Held Stocks (that are in target)
        # But wait, we want to rebalance existing ones too?
        # Simple approach: Sell everything and buy Top K? No, that causes turnover.
        # Efficient approach: Calculate target amount.
        
        # Let's use a simpler logic for "Turnover Reduction":
        # Only sell what is NOT in Top K.
        # Only buy what IS in Top K but not held.
        # Do not rebalance weights of existing holdings (buffer).
        # This minimizes turnover.
        
        # held_target_stocks = [s for s in current_stock_list if s in target_stocks]
        # new_target_stocks = [s for s in target_stocks if s not in current_stock_list]
        
        # But we need to use the cash.
        # If we just hold, we drift.
        # Let's stick to "Equal Weight on Top K". This generates some turnover (rebalancing) but less than "Swap 1 stock per day".
        # Actually, TopkDropoutStrategy swaps 1 stock. SimpleTopk swaps ONLY when rank changes > K.
        # So "Equal Weight Top K" is the standard baseline.
        
        # Re-calc cash and value
        current_value = current_temp.calculate_value() # This uses current prices?
        # We can just allocate cash to new buys?
        # Let's try to fully rebalance for correctness, but the "Swap" condition is only on rank.
        
        target_value_per_stock = current_value * self.risk_degree / self.topk
        
        for code in target_stocks:
            if not self.trade_exchange.is_stock_tradable(stock_id=code, start_time=trade_start_time, end_time=trade_end_time, direction=OrderDir.BUY):
                continue
            
            # Current holding
            current_amount = current_temp.get_stock_amount(code=code)
            current_price = self.trade_exchange.get_deal_price(stock_id=code, start_time=trade_start_time, end_time=trade_end_time, direction=OrderDir.BUY)
            current_holding_value = current_amount * current_price
            
            # Gap
            target_amount = target_value_per_stock / current_price
            msg_amount = target_amount - current_amount
            
            # Determine direction
            if msg_amount > 0:
                # Buy
                # Round to lot size
                factor = self.trade_exchange.get_factor(stock_id=code, start_time=trade_start_time, end_time=trade_end_time)
                buy_amount = self.trade_exchange.round_amount_by_trade_unit(msg_amount, factor)
                if buy_amount > 0:
                    buy_order = Order(
                        stock_id=code,
                        amount=buy_amount,
                        start_time=trade_start_time,
                        end_time=trade_end_time,
                        direction=OrderDir.BUY
                    )
                    buy_order_list.append(buy_order)
            elif msg_amount < 0:
                 # Sell (Rebalance down) - optional, can skip to reduce turnover further?
                 # Experiment 3 goal is turnover < 500%. Full rebalance might be high.
                 # Let's only sell if we need cash?
                 # No, standard TopK implies rebalancing.
                 # Let's do full rebalance first.
                 sell_amount = abs(msg_amount)
                 sell_order = Order(
                    stock_id=code,
                    amount=sell_amount,
                    start_time=trade_start_time,
                    end_time=trade_end_time,
                    direction=Order.SELL
                 )
                 sell_order_list.append(sell_order)

        return TradeDecisionWO(sell_order_list + buy_order_list, self)

class BacktestEngine:
    def __init__(self, pred):
        self.pred = pred

    def run(self):
        print("\nRunning Backtest (Experiment 3: Custom SimpleTopkStrategy)...")
        # Strategy Config
        STRATEGY_CONFIG = {
            "topk": 3,
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
