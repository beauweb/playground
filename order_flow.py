import pandas as pd
import numpy as np
from collections import deque

class OrderFlowAnalyzer:
    def __init__(self, depth_limit=10):
        """Initialize order flow analyzer"""
        self.depth_limit = depth_limit
        self.order_history = deque(maxlen=1000)
        
    def analyze_order_book(self, bids, asks):
        """Analyze current order book state"""
        total_bid_volume = sum(bid[1] for bid in bids[:self.depth_limit])
        total_ask_volume = sum(ask[1] for ask in asks[:self.depth_limit])
        
        # Calculate buying/selling pressure
        buy_pressure = total_bid_volume / (total_bid_volume + total_ask_volume)
        sell_pressure = 1 - buy_pressure
        
        # Identify large orders (whales)
        large_bids = [bid for bid in bids if bid[1] > total_bid_volume * 0.1]
        large_asks = [ask for ask in asks if ask[1] > total_ask_volume * 0.1]
        
        # Calculate order book imbalance
        imbalance = (total_bid_volume - total_ask_volume) / (total_bid_volume + total_ask_volume)
        
        return {
            'buy_pressure': buy_pressure,
            'sell_pressure': sell_pressure,
            'large_orders': {
                'bids': large_bids,
                'asks': large_asks
            },
            'imbalance': imbalance
        }
        
    def analyze_trades(self, trades):
        """Analyze recent trades"""
        buy_volume = sum(trade['amount'] for trade in trades if trade['side'] == 'buy')
        sell_volume = sum(trade['amount'] for trade in trades if trade['side'] == 'sell')
        
        # Calculate trade flow
        total_volume = buy_volume + sell_volume
        buy_ratio = buy_volume / total_volume if total_volume > 0 else 0.5
        
        # Identify large trades
        avg_trade_size = total_volume / len(trades) if trades else 0
        large_trades = [trade for trade in trades if trade['amount'] > avg_trade_size * 3]
        
        return {
            'buy_ratio': buy_ratio,
            'sell_ratio': 1 - buy_ratio,
            'large_trades': large_trades,
            'average_trade_size': avg_trade_size
        }
        
    def get_liquidity_analysis(self, bids, asks):
        """Analyze market liquidity"""
        # Calculate bid-ask spread
        best_bid = max(bid[0] for bid in bids)
        best_ask = min(ask[0] for ask in asks)
        spread = (best_ask - best_bid) / best_bid
        
        # Calculate liquidity depth
        depth_prices = {
            'bid_1pct': sum(bid[1] for bid in bids if bid[0] > best_bid * 0.99),
            'bid_2pct': sum(bid[1] for bid in bids if bid[0] > best_bid * 0.98),
            'ask_1pct': sum(ask[1] for ask in asks if ask[0] < best_ask * 1.01),
            'ask_2pct': sum(ask[1] for ask in asks if ask[0] < best_ask * 1.02)
        }
        
        return {
            'spread': spread,
            'depth': depth_prices,
            'liquidity_score': 1.0 / (spread * 100)  # Higher score = better liquidity
        }
        
    def get_market_impact(self, order_size, bids, asks):
        """Estimate market impact of a given order size"""
        impact = {
            'buy': 0.0,
            'sell': 0.0
        }
        
        # Calculate price impact for buy order
        remaining_size = order_size
        last_price = asks[0][0]
        for ask in asks:
            if remaining_size <= 0:
                break
            if remaining_size >= ask[1]:
                impact['buy'] = (ask[0] - last_price) / last_price
            remaining_size -= ask[1]
            
        # Calculate price impact for sell order
        remaining_size = order_size
        last_price = bids[0][0]
        for bid in bids:
            if remaining_size <= 0:
                break
            if remaining_size >= bid[1]:
                impact['sell'] = (last_price - bid[0]) / last_price
            remaining_size -= bid[1]
            
        return impact