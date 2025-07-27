"""Strategy definitions and signal generation.

This module brings together technical indicators and candlestick
patterns to generate buy/sell signals.  Strategies are configurable
via a dictionary that enables or disables specific patterns and
indicators and sets thresholds for indicators such as RSI.  The main
entry point is the :class:`Strategy` class.
"""

from __future__ import annotations

import pandas as pd
from typing import Dict, Optional

from . import indicators as ind
from . import patterns as pat


class Strategy:
    """Trading strategy composed of patterns and indicators.

    The strategy uses a configuration dictionary (usually loaded from
    the YAML config) to determine which patterns and indicators to
    compute and which thresholds to apply.  It exposes a
    :meth:`generate_signals` method that takes a DataFrame of
    historical price data and returns a DataFrame of signals aligned
    with the input index.
    """

    def __init__(self, config: Dict[str, any]):
        self.pattern_config = config.get("patterns", {})
        self.indicator_config = config.get("indicators", {})
        self.thresholds = config.get("thresholds", {})

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute the enabled technical indicators and append them to the DataFrame.

        Args:
            df: DataFrame with OHLCV data.

        Returns:
            New DataFrame with additional indicator columns.
        """
        result = df.copy()
        close = result["close"]
        high = result["high"]
        low = result["low"]

        # Moving averages
        if self.indicator_config.get("sma_50"):
            result["sma_50"] = ind.sma(close, 50)
        if self.indicator_config.get("sma_100"):
            result["sma_100"] = ind.sma(close, 100)
        if self.indicator_config.get("sma_200"):
            result["sma_200"] = ind.sma(close, 200)
        if self.indicator_config.get("sma_20"):
            result["sma_20"] = ind.sma(close, 20)
        if self.indicator_config.get("ema_50"):
            result["ema_50"] = ind.ema(close, 50)
        if self.indicator_config.get("ema_100"):
            result["ema_100"] = ind.ema(close, 100)
        if self.indicator_config.get("ema_200"):
            result["ema_200"] = ind.ema(close, 200)
            
        # RSI
        if self.indicator_config.get("rsi"):
            window = self.indicator_config.get("rsi_window", 14)
            result["rsi"] = ind.rsi(close, window)
            
        # Enhanced trend analysis
        if self.indicator_config.get("trend_strength"):
            short_window = self.indicator_config.get("trend_short_window", 20)
            long_window = self.indicator_config.get("trend_long_window", 50)
            result["trend_strength"] = ind.trend_strength(result, short_window, long_window)
            
        if self.indicator_config.get("weekly_trend"):
            weekly_analysis = ind.weekly_trend_analysis(result)
            for col in weekly_analysis.columns:
                result[f"weekly_{col}"] = weekly_analysis[col]
                
        if self.indicator_config.get("monthly_trend"):
            monthly_analysis = ind.monthly_trend_analysis(result)
            for col in monthly_analysis.columns:
                result[f"monthly_{col}"] = monthly_analysis[col]
                
        # Enhanced volume analysis
        if self.indicator_config.get("volume_analysis"):
            vol_window = self.indicator_config.get("volume_window", 20)
            volume_analysis = ind.volume_trend_analysis(result, vol_window)
            for col in volume_analysis.columns:
                result[f"vol_{col}"] = volume_analysis[col]
        elif self.indicator_config.get("volume"):
            # Basic volume analysis (backward compatibility)
            result["vol_sma_20"] = ind.sma(result["volume"], 20)
            
        # Advanced support/resistance
        if self.indicator_config.get("advanced_support_resistance"):
            sr_window = self.indicator_config.get("sr_window", 20)
            min_touches = self.indicator_config.get("sr_min_touches", 2)
            sr_analysis = ind.advanced_support_resistance(result, sr_window, min_touches)
            for col in sr_analysis.columns:
                result[f"sr_{col}"] = sr_analysis[col]
                
        # Support/resistance – pivot points on daily data (resampled)
        if self.indicator_config.get("support_resistance"):
            # Resample to daily using last values and compute pivot
            daily = result.resample("1D").agg({
                "high": "max",
                "low": "min",
                "close": "last",
            })
            pivots = ind.pivot_points(daily)
            # Forward fill to intraday index
            pivots = pivots.reindex(result.index, method="ffill")
            for col in pivots.columns:
                result[f"pivot_{col}"] = pivots[col]
                
        # Fibonacci levels – compute once per series
        if self.indicator_config.get("fibonacci"):
            levels = ind.fibonacci_levels(close)
            for label, value in levels.items():
                result[f"fib_{label}"] = value
                
        # Higher lows / lower highs
        if self.indicator_config.get("higher_lows"):
            result["higher_lows"] = ind.higher_lows(low)
        if self.indicator_config.get("lower_highs"):
            result["lower_highs"] = ind.lower_highs(high)
            
        # Golden/death crosses: need moving averages; ensure they exist
        if self.indicator_config.get("golden_cross"):
            if "sma_50" in result.columns and "sma_200" in result.columns:
                result["golden_cross"] = ind.golden_cross(result["sma_50"], result["sma_200"])
            elif "ema_50" in result.columns and "ema_200" in result.columns:
                result["golden_cross"] = ind.golden_cross(result["ema_50"], result["ema_200"])
            else:
                result["golden_cross"] = False
        if self.indicator_config.get("death_cross"):
            if "sma_50" in result.columns and "sma_200" in result.columns:
                result["death_cross"] = ind.death_cross(result["sma_50"], result["sma_200"])
            elif "ema_50" in result.columns and "ema_200" in result.columns:
                result["death_cross"] = ind.death_cross(result["ema_50"], result["ema_200"])
            else:
                result["death_cross"] = False
        return result

    def compute_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute the enabled candlestick patterns.

        Args:
            df: DataFrame with OHLC data.

        Returns:
            DataFrame with pattern indicator columns.
        """
        result = pd.DataFrame(index=df.index)
        if self.pattern_config.get("bullish_engulfing"):
            result["bullish_engulfing"] = pat.detect_bullish_engulfing(df)
        if self.pattern_config.get("bearish_engulfing"):
            result["bearish_engulfing"] = pat.detect_bearish_engulfing(df)
        if self.pattern_config.get("gap_up"):
            result["gap_up"] = pat.detect_gap_up(df)
        if self.pattern_config.get("gap_down"):
            result["gap_down"] = pat.detect_gap_down(df)
        if self.pattern_config.get("hammer"):
            result["hammer"] = pat.detect_hammer(df)
        if self.pattern_config.get("shooting_star"):
            result["shooting_star"] = pat.detect_shooting_star(df)
        if self.pattern_config.get("double_top"):
            result["double_top"] = pat.detect_double_top(df)
        if self.pattern_config.get("double_bottom"):
            result["double_bottom"] = pat.detect_double_bottom(df)
        if self.pattern_config.get("triple_top"):
            result["triple_top"] = pat.detect_triple_top(df)
        if self.pattern_config.get("triple_bottom"):
            result["triple_bottom"] = pat.detect_triple_bottom(df)
        if self.pattern_config.get("cup_and_handle"):
            result["cup_and_handle"] = pat.detect_cup_and_handle(df)
        if self.pattern_config.get("rising_wedge"):
            result["rising_wedge"] = pat.detect_rising_wedge(df)
            
        # New Doji patterns
        if self.pattern_config.get("doji"):
            tolerance = self.pattern_config.get("doji_tolerance", 0.001)
            result["doji"] = pat.detect_doji(df, tolerance)
        if self.pattern_config.get("dragonfly_doji"):
            tolerance = self.pattern_config.get("doji_tolerance", 0.001)
            result["dragonfly_doji"] = pat.detect_dragonfly_doji(df, tolerance)
        if self.pattern_config.get("gravestone_doji"):
            tolerance = self.pattern_config.get("doji_tolerance", 0.001)
            result["gravestone_doji"] = pat.detect_gravestone_doji(df, tolerance)
        if self.pattern_config.get("spinning_top"):
            result["spinning_top"] = pat.detect_spinning_top(df)
            
        if self.pattern_config.get("higher_lows"):
            # Higher lows and lower highs can be considered patterns as well
            result["higher_lows"] = ind.higher_lows(df["low"])
        if self.pattern_config.get("lower_highs"):
            result["lower_highs"] = ind.lower_highs(df["high"])
        return result

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals from price data.

        Returns a DataFrame with ``signal`` column taking values 1 (buy),
        -1 (sell) or 0 (hold).  Additional diagnostic columns are
        included for debugging (patterns and indicator values).
        """
        # Compute indicators and patterns
        indicators = self.compute_indicators(df)
        patterns = self.compute_patterns(df)
        signals = pd.DataFrame(index=df.index)
        signals = pd.concat([indicators, patterns], axis=1)

        # Initialise signal column with zeros
        signal = pd.Series(0, index=signals.index, dtype=int)

        # RSI thresholds
        rsi_overbought = self.thresholds.get("rsi_overbought", 70)
        rsi_oversold = self.thresholds.get("rsi_oversold", 30)

        # Calculate signal strength (0-100 scale)
        buy_strength = pd.Series(0.0, index=signals.index)
        sell_strength = pd.Series(0.0, index=signals.index)

        # RSI conditions (weight: 25)
        if "rsi" in signals.columns:
            # Stronger signal when RSI is more extreme
            rsi_buy_strength = ((rsi_oversold - signals["rsi"]) / rsi_oversold * 25).clip(0, 25)
            rsi_sell_strength = ((signals["rsi"] - rsi_overbought) / (100 - rsi_overbought) * 25).clip(0, 25)
            buy_strength += rsi_buy_strength.fillna(0)
            sell_strength += rsi_sell_strength.fillna(0)
            
        # Trend strength conditions (weight: 20)
        if "trend_strength" in signals.columns:
            trend_buy_threshold = self.thresholds.get("trend_strength_buy", 20)
            trend_sell_threshold = self.thresholds.get("trend_strength_sell", -20)
            
            trend_buy_strength = (signals["trend_strength"] / 100 * 20).clip(0, 20)
            trend_sell_strength = (-signals["trend_strength"] / 100 * 20).clip(0, 20)
            buy_strength += trend_buy_strength.fillna(0)
            sell_strength += trend_sell_strength.fillna(0)
            
        # Weekly trend conditions (weight: 15)
        if "weekly_trend_direction" in signals.columns:
            weekly_buy = (signals["weekly_trend_direction"] > 0) * 15
            weekly_sell = (signals["weekly_trend_direction"] < 0) * 15
            buy_strength += weekly_buy.fillna(0)
            sell_strength += weekly_sell.fillna(0)
            
        # Volume confirmation (weight: 10)
        if "vol_vol_price_confirmation" in signals.columns:
            vol_buy = (signals["vol_vol_price_confirmation"] > 0) * 10
            vol_sell = (signals["vol_vol_price_confirmation"] < 0) * 10
            buy_strength += vol_buy.fillna(0)
            sell_strength += vol_sell.fillna(0)
            
        # Price relative to SMA-20 (weight: 10)
        if "sma_20" in signals.columns:
            price_above_sma = (signals["close"] > signals["sma_20"]) * 10
            price_below_sma = (signals["close"] < signals["sma_20"]) * 10
            buy_strength += price_above_sma.fillna(0)
            sell_strength += price_below_sma.fillna(0)
            
        # Pattern signals (weight: 15)
        bullish_cols = [col for col in patterns.columns if any(keyword in col for keyword in 
                       ["bullish", "gap_up", "hammer", "double_bottom", "triple_bottom", 
                        "cup_and_handle", "higher_lows", "dragonfly_doji"])]
        if bullish_cols:
            pattern_buy = patterns[bullish_cols].any(axis=1) * 15
            buy_strength += pattern_buy.fillna(0)
            
        bearish_cols = [col for col in patterns.columns if any(keyword in col for keyword in 
                       ["bearish", "gap_down", "shooting_star", "double_top", "triple_top", 
                        "rising_wedge", "lower_highs", "gravestone_doji"])]
        if bearish_cols:
            pattern_sell = patterns[bearish_cols].any(axis=1) * 15
            sell_strength += pattern_sell.fillna(0)
            
        # Golden/Death cross signals (weight: 5)
        if "golden_cross" in signals.columns:
            golden_cross_signal = signals["golden_cross"] * 5
            buy_strength += golden_cross_signal.fillna(0)
            
        if "death_cross" in signals.columns:
            death_cross_signal = signals["death_cross"] * 5
            sell_strength += death_cross_signal.fillna(0)

        # Signal thresholds
        buy_threshold = self.thresholds.get("buy_signal_threshold", 50)
        sell_threshold = self.thresholds.get("sell_signal_threshold", 50)
        
        # Generate final signals with conflict resolution
        strong_buy = buy_strength >= buy_threshold
        strong_sell = sell_strength >= sell_threshold
        
        # Conflict resolution: stronger signal wins
        buy_dominance = buy_strength > sell_strength
        sell_dominance = sell_strength > buy_strength
        
        # Final signal assignment
        signal = pd.Series(0, index=signals.index, dtype=int)
        signal = signal.where(~(strong_buy & buy_dominance), 1)  # Buy signal
        signal = signal.where(~(strong_sell & sell_dominance), -1)  # Sell signal
        
        # Add diagnostic columns
        signals["signal"] = signal
        signals["buy_strength"] = buy_strength
        signals["sell_strength"] = sell_strength
        signals["signal_confidence"] = (buy_strength - sell_strength).abs()
        
        return signals
