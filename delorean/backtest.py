from typing import List, Tuple, Dict, Any, Optional
from qlib.contrib.strategy.signal_strategy import BaseSignalStrategy
from qlib.backtest import backtest as qlib_backtest
from qlib.backtest import executor as qlib_executor
from qlib.backtest.decision import Order, OrderDir, TradeDecisionWO
from .config import TEST_START_TIME, END_TIME, BENCHMARK
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
    def __init__(self, topk: int = 4, risk_degree: float = 0.95, drop_rate: float = 0.96, n_drop: int = 1, market_regime: pd.Series = None, vol_feature: pd.Series = None, bench_feature: pd.DataFrame = None, **kwargs: Any):
        super().__init__(risk_degree=risk_degree, **kwargs)
        self.topk = topk
        self.drop_rate = drop_rate
        self.n_drop = n_drop
        self.market_regime = market_regime
        self.vol_feature = vol_feature # Index: (datetime, instrument), Value: VOL20
        self.bench_feature = bench_feature # Index: datetime, Columns: close, ma60
        self.last_risk_degree = risk_degree # State for Hysteresis

    def generate_trade_decision(self, execute_result: Any = None) -> TradeDecisionWO:
        """
        Generate trade orders with turnover control and market regime filter.
        """
        # 1. Get Trading Step and Time
        trade_step = self.trade_calendar.get_trade_step()
        trade_start_time, trade_end_time = self.trade_calendar.get_step_time(trade_step)

        # 0. Market Regime & Dynamic Exposure (with Hysteresis)
        current_risk_degree = self.last_risk_degree # Default to last state
        
        # If market_regime provided (Boolean Series) - Legacy/Override
        if self.market_regime is not None:
             try:
                is_bull = self.market_regime.loc[trade_start_time] if trade_start_time in self.market_regime.index else True
             except:
                is_bull = True
             
             if not is_bull:
                 # Standard Regime Filter: Sell All
                 return TradeDecisionWO(self._generate_sell_orders(self.trade_position, pd.Index([]), trade_start_time, trade_end_time), self)

        # [NEW] Dynamic Exposure (Trend Strength) with Hysteresis
        if self.bench_feature is not None:
             try:
                 # Check if we have data for this date
                 if trade_start_time in self.bench_feature.index:
                     row = self.bench_feature.loc[trade_start_time]
                     close = row['close']
                     ma60 = row['ma60']
                     
                     trend_strength = (close - ma60) / ma60
                     
                     # Hysteresis Logic
                     # States: High (0.99), Med (0.80), Off (0.0)
                     # Transitions with buffers
                     
                     if current_risk_degree >= 0.99:
                         # In High -> Downgrade to Med if < 1%
                         if trend_strength < 0.01:
                             current_risk_degree = 0.80
                     elif current_risk_degree >= 0.80:
                         # In Med -> Upgrade to High if > 2%
                         if trend_strength > 0.02:
                             current_risk_degree = 0.99
                         # In Med -> Downgrade to Off if < -1% (Buffer below 0)
                         elif trend_strength < -0.01:
                             current_risk_degree = 0.0
                     else:
                         # In Off -> Upgrade to Med if > 0%
                         if trend_strength > 0:
                             current_risk_degree = 0.80

                     # Bear (Below MA60) -> 0% (Cash) - Implicitly handled by transition to Off
                     if current_risk_degree == 0.0:
                          self.last_risk_degree = 0.0
                          return TradeDecisionWO(self._generate_sell_orders(self.trade_position, pd.Index([]), trade_start_time, trade_end_time), self)

             except Exception as e:
                 pass # Fallback to default
        
        # Update State
        self.last_risk_degree = current_risk_degree
        
        # 0.5 Turnover control: Probabilistic Retention
        # If random < drop_rate, we SKIP trading this step (hold existing)
        # Only applies in Bull Market (if we are here)
        
        # [FIX] Do NOT skip if we need to ENTER the market (current holdings empty but bullish)
        current_holdings_list = self.trade_position.get_stock_list()
        should_force_trade = (len(current_holdings_list) == 0) and (current_risk_degree > 0)
        
        if (not should_force_trade) and (random.random() < self.drop_rate):
            # IMPORTANT: If using Dynamic Exposure, we might need to REBALANCE cash even if we skip turnover rotation.
            # But simplistic approach: if skip, do nothing.
            return TradeDecisionWO([], self)
        
        # 2. Get Scores
        pred_score = self._get_pred_scores(trade_step)
        if pred_score is None:
            return TradeDecisionWO([], self)
            
        # 3. Determine Target Stocks (Top K)
        # We need the full sorted list to find best replacements if needed
        sorted_score = pred_score.sort_values(ascending=False)
        target_stocks = sorted_score.head(self.topk).index
        
        # 4. Current Position Snapshot
        current_temp = copy.deepcopy(self.trade_position)
        current_stock_list = current_temp.get_stock_list()
        
        # 5. Sell Logic with n_drop limit
        # Identify stocks we hold that are NOT in target
        hold_not_in_target = [code for code in current_stock_list if code not in target_stocks]

        # Get scores for held stocks
        score_map = pred_score.to_dict()
        
        # Sort held_not_in_target by score (ascending: drop worst)
        hold_not_in_target.sort(key=lambda x: score_map.get(x, -9999))
        
        # Select ones to actually drop (limit by n_drop)
        to_drop = hold_not_in_target[:self.n_drop]
        
        sell_order_list = []
        for code in to_drop:
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
                # Update temp position to free cash
                self.trade_exchange.deal_order(sell_order, position=current_temp)

        # 6. Buy Logic
        # We want to fill up to `topk`.
        
        buy_order_list = []
        sell_order_rebalance_list = []
        
        current_holdings_after_sell = current_temp.get_stock_list()
        
        # Merge current holdings + new buys to form "Target Portfolio" for today
        slots_available = self.topk - len(current_holdings_after_sell)
        
        final_target_stocks = list(current_holdings_after_sell)
        
        if slots_available > 0:
            # Pick best candidates from target_stocks not currently held
            candidates = [s for s in target_stocks if s not in current_holdings_after_sell]
            # Since target_stocks is already topk sorted, just take top `slots_available`
            to_buy = candidates[:slots_available]
            final_target_stocks.extend(to_buy)
        else:
            to_buy = []

        # [NEW] Volatility Targeting (Risk Parity)
        # Calculate weights for final_target_stocks
        # Formula: w_i = (1/vol_i) / sum(1/vol_j)
        
        target_weights = {} # code -> weight
        
        if self.vol_feature is not None:
             # Get volatility for these stocks on this date
             inv_vols = {}
             sum_inv_vol = 0.0
             
             for code in final_target_stocks:
                 try:
                     vol = self.vol_feature.loc[(trade_start_time, code)]
                     if pd.isna(vol) or vol <= 0: vol = 1.0 # Fallback
                 except:
                     vol = 1.0 # Fallback
                     
                 inv_vol = 1.0 / vol
                 inv_vols[code] = inv_vol
                 sum_inv_vol += inv_vol
                 
             if sum_inv_vol > 0:
                 for code in final_target_stocks:
                     target_weights[code] = inv_vols[code] / sum_inv_vol
             else:
                 # Fallback to EW
                 for code in final_target_stocks:
                     target_weights[code] = 1.0 / len(final_target_stocks)
        else:
             # Equal Weight
             for code in final_target_stocks:
                 target_weights[code] = 1.0 / len(final_target_stocks)

        # Execute Buys / Rebalance
        # Calculate target value
        current_total_value = current_temp.calculate_value()
        
        # [OPTIMIZATION] Buffer Rebalancing
        # Threshold: 2% of total portfolio value
        buffer_threshold = current_total_value * 0.02
        
        for code in final_target_stocks:
            # Buy/Sell to reach target weight
            target_val = current_total_value * current_risk_degree * target_weights[code]
            
            current_amount = current_temp.get_stock_amount(code=code)
            
             # Check Price
            if not self.trade_exchange.is_stock_tradable(stock_id=code, start_time=trade_start_time, end_time=trade_end_time, direction=OrderDir.BUY):
                 continue
            current_price = self.trade_exchange.get_deal_price(stock_id=code, start_time=trade_start_time, end_time=trade_end_time, direction=OrderDir.BUY)
            
            target_amount = target_val / current_price
            diff_amount = target_amount - current_amount
            diff_val = diff_amount * current_price
            
            # Skip if deviation is within buffer (unless it is a new buy which effectively has diff_val = target_val)
            # Actually diff_val is the absolute dollar amount we want to trade.
            if abs(diff_val) < buffer_threshold:
                continue
            
            factor = self.trade_exchange.get_factor(stock_id=code, start_time=trade_start_time, end_time=trade_end_time)
            
            if diff_amount > 0:
                # Buy
                buy_amount = self.trade_exchange.round_amount_by_trade_unit(diff_amount, factor)
                if buy_amount > 0:
                     buy_order = Order(
                        stock_id=code,
                        amount=buy_amount,
                        start_time=trade_start_time,
                        end_time=trade_end_time,
                        direction=OrderDir.BUY
                    )
                     buy_order_list.append(buy_order)
            elif diff_amount < 0:
                # Sell (Rebalance Down)
                sell_amount = self.trade_exchange.round_amount_by_trade_unit(abs(diff_amount), factor)
                if sell_amount > 0:
                     sell_order = Order(
                        stock_id=code,
                        amount=sell_amount,
                        start_time=trade_start_time,
                        end_time=trade_end_time,
                        direction=OrderDir.SELL
                    )
                     sell_order_rebalance_list.append(sell_order)

        return TradeDecisionWO(sell_order_list + sell_order_rebalance_list + buy_order_list, self)

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

    def run(self, topk: int = 3, market_regime: pd.Series = None, **kwargs: Any) -> Tuple[pd.DataFrame, Dict[Any, Any]]:
        """
        Run the backtest simulation.

        Args:
            topk (int): Number of stocks to hold in the TopK strategy.
            market_regime (pd.Series): Boolean series (True=Bull, False=Bear) to filter trades.
            **kwargs: Additional strategy parameters (drop_rate, n_drop).

        Returns:
            Tuple[pd.DataFrame, dict]: A tuple containing the backtest report DataFrame 
            and a dictionary of position details.
        """
        print(f"\nRunning Backtest (Custom SimpleTopkStrategy, TopK={topk}, RegimeFilter={'On' if market_regime is not None else 'Off'}, Params={kwargs})...")
        # Strategy Config
        STRATEGY_CONFIG = {
            "topk": topk,
            "risk_degree": 0.95,
            "signal": self.pred,
            "market_regime": market_regime,
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

        # Determine Backtest End Time
        # The backtest engine's calendar logic requires the 'next' step to exist to define the interval.
        # If we run until the absolute last available date, get_step_time(i) tries to access i+1.
        # So we must stop one step before the end of the data/calendar.
        
        valid_dates = self.pred.index.get_level_values('datetime').unique().sort_values()
        if len(valid_dates) > 1:
            # Use the second to last date as the safe end_time for simulation
            # This ensures 'next day' exists in the calendar (which is the last date)
            data_end_time = valid_dates[-2]
        else:
            data_end_time = valid_dates[-1]

        portfolio_metric_dict, indicator_dict = qlib_backtest(
            executor=executor_obj,
            strategy=strategy_obj,
            start_time=TEST_START_TIME,
            end_time=data_end_time,
            account=1000000,
            benchmark=BENCHMARK,
        )
        
        # Extract report and positions (key is usually '1day' for daily frequency)
        # portfolio_metric_dict mapping: freq -> (report_df, positions_dict)
        report, positions = list(portfolio_metric_dict.values())[0]
        
        return report, positions
