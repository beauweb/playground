import pandas as pd
import numpy as np
from typing import List, Dict, Tuple

class MTFAnalyzer:
    def __init__(self, timeframes: List[str] = None):
        """Initialize Multi-Timeframe analyzer"""
        self.timeframes = timeframes or ['1m', '5m', '15m', '1h', '4h', '1d']
        self.indicators = {}
        
    def calculate_mtf_indicators(self, dfs: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
        """Calculate indicators for multiple timeframes"""
        results = {}
        for timeframe, df in dfs.items():
            results[timeframe] = self._calculate_indicators(df)
        return results
    
    def _calculate_indicators(self, df: pd.DataFrame) -> Dict:
        """Calculate technical indicators for a single timeframe"""
        # Calculate EMAs
        df['ema9'] = df['close'].ewm(span=9, adjust=False).mean()
        df['ema21'] = df['close'].ewm(span=21, adjust=False).mean()
        df['ema55'] = df['close'].ewm(span=55, adjust=False).mean()
        
        # Calculate RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Calculate MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        
        # Return last values
        return {
            'ema9': df['ema9'].iloc[-1],
            'ema21': df['ema21'].iloc[-1],
            'ema55': df['ema55'].iloc[-1],
            'rsi': df['rsi'].iloc[-1],
            'macd': df['macd'].iloc[-1],
            'macd_signal': df['signal'].iloc[-1]
        }
    
    def get_mtf_trend(self, indicators: Dict[str, Dict]) -> Dict[str, str]:
        """Determine trend for each timeframe"""
        trends = {}
        for timeframe, vals in indicators.items():
            # EMA trend
            ema_trend = 'bullish' if (vals['ema9'] > vals['ema21'] > vals['ema55']) else \
                       'bearish' if (vals['ema9'] < vals['ema21'] < vals['ema55']) else \
                       'neutral'
            
            # RSI trend
            rsi_trend = 'bullish' if vals['rsi'] > 60 else \
                       'bearish' if vals['rsi'] < 40 else \
                       'neutral'
            
            # MACD trend
            macd_trend = 'bullish' if (vals['macd'] > vals['macd_signal'] and vals['macd'] > 0) else \
                        'bearish' if (vals['macd'] < vals['macd_signal'] and vals['macd'] < 0) else \
                        'neutral'
            
            # Combined trend
            trends[timeframe] = self._combine_trends([ema_trend, rsi_trend, macd_trend])
            
        return trends
    
    def _combine_trends(self, trends: List[str]) -> str:
        """Combine multiple trend signals into one"""
        bull_count = trends.count('bullish')
        bear_count = trends.count('bearish')
        
        if bull_count > bear_count and bull_count >= len(trends) / 2:
            return 'bullish'
        elif bear_count > bull_count and bear_count >= len(trends) / 2:
            return 'bearish'
        else:
            return 'neutral'
    
    def get_trend_strength(self, trends: Dict[str, str]) -> float:
        """Calculate overall trend strength across timeframes"""
        weights = {
            '1m': 0.05,
            '5m': 0.10,
            '15m': 0.15,
            '1h': 0.25,
            '4h': 0.25,
            '1d': 0.20
        }
        
        strength = 0
        for timeframe, trend in trends.items():
            if timeframe in weights:
                if trend == 'bullish':
                    strength += weights[timeframe]
                elif trend == 'bearish':
                    strength -= weights[timeframe]
                    
        return strength  # Range: -1 to 1
    
    def get_trend_alignment(self, trends: Dict[str, str]) -> Dict:
        """Check if trends are aligned across timeframes"""
        # Count trends
        trend_counts = {
            'bullish': 0,
            'bearish': 0,
            'neutral': 0
        }
        for trend in trends.values():
            trend_counts[trend] += 1
            
        # Calculate alignment percentage
        max_aligned = max(trend_counts.values())
        alignment = max_aligned / len(trends)
        
        # Determine dominant trend
        dominant_trend = max(trend_counts.items(), key=lambda x: x[1])[0]
        
        return {
            'alignment': alignment,  # Range: 0 to 1
            'dominant_trend': dominant_trend,
            'counts': trend_counts
        }