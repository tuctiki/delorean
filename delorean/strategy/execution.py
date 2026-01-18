from typing import List, Dict, Any, Optional
import copy
import logging
from qlib.backtest.decision import Order, OrderDir

class ExecutionModel:
    """
    Handles order generation with turnover control (buffer).
    """
    def __init__(self, topk: int = 5, buffer: int = 2, n_drop: int = 2, rebalance_threshold: float = 0.05):
        self.topk = topk
        self.buffer = buffer
        self.n_drop = n_drop
        self.rebalance_threshold = rebalance_threshold  # Default increased to 5%
        
    def generate_orders(self, 
                        current_pos: Any, 
                        pred_score: Any, 
                        target_weights: Dict[str, float], 
                        trade_exchange: Any,
                        trade_start_time: Any,
                        trade_end_time: Any) -> List[Order]:
        
        # 1. Identify Sells (Buffer Logic)
        current_stock_list = current_pos.get_stock_list()
        ranks = pred_score.rank(ascending=False)
        
        to_drop = []
        for code in current_stock_list:
            rank = ranks.get(code, 99999)
            if rank > (self.topk + self.buffer):
                to_drop.append(code)
                
        # n_drop limit
        to_drop.sort(key=lambda x: ranks.get(x, 99999), reverse=True)
        to_drop = to_drop[:self.n_drop]
        
        # 2. Generate Sell Orders
        orders = []
        current_temp = copy.deepcopy(current_pos)
        
        for code in to_drop:
            if not trade_exchange.is_stock_tradable(stock_id=code, start_time=trade_start_time, end_time=trade_end_time, direction=OrderDir.SELL):
                continue
                
            sell_amount = current_temp.get_stock_amount(code=code)
            sell_order = Order(
                stock_id=code,
                amount=sell_amount,
                start_time=trade_start_time,
                end_time=trade_end_time,
                direction=Order.SELL
            )
            if trade_exchange.check_order(sell_order):
                orders.append(sell_order)
                trade_exchange.deal_order(sell_order, position=current_temp)
                
        # 3. Identify Buys
        # Current holdings after sell
        current_holdings = current_temp.get_stock_list()
        slots = self.topk - len(current_holdings)
        
        # Select best available candidates not held
        sorted_score = pred_score.sort_values(ascending=False)
        candidates = sorted_score.index.tolist()
        
        final_targets = list(current_holdings)
        if slots > 0:
            count = 0
            for code in candidates:
                if count >= slots: break
                if code not in current_holdings:
                    final_targets.append(code)
                    count += 1
                    
        # 4. Generate Buy/Rebalance Orders
        current_total_value = current_temp.calculate_value()
        buffer_val_threshold = current_total_value * self.rebalance_threshold  # Use configurable threshold
        
        for code in final_targets:
            if code not in target_weights: continue # Should generally overlap
            
            target_val = current_total_value * target_weights[code]
            current_amount = current_temp.get_stock_amount(code=code)
            
            if not trade_exchange.is_stock_tradable(stock_id=code, start_time=trade_start_time, end_time=trade_end_time, direction=OrderDir.BUY):
                continue
                
            price = trade_exchange.get_deal_price(stock_id=code, start_time=trade_start_time, end_time=trade_end_time, direction=OrderDir.BUY)
            target_amount = target_val / price
            
            diff_amount = target_amount - current_amount
            diff_val = diff_amount * price
            
            if abs(diff_val) < buffer_val_threshold:
                continue
                
            factor = trade_exchange.get_factor(stock_id=code, start_time=trade_start_time, end_time=trade_end_time)
            
            if diff_amount > 0:
                buy_amt = trade_exchange.round_amount_by_trade_unit(diff_amount, factor)
                if buy_amt > 0:
                    orders.append(Order(stock_id=code, amount=buy_amt, start_time=trade_start_time, end_time=trade_end_time, direction=OrderDir.BUY))
            else:
                sell_amt = trade_exchange.round_amount_by_trade_unit(abs(diff_amount), factor)
                if sell_amt > 0:
                    orders.append(Order(stock_id=code, amount=sell_amt, start_time=trade_start_time, end_time=trade_end_time, direction=OrderDir.SELL))
                    
        return orders
