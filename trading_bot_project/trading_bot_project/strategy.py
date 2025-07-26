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
        # Volume – simple moving average of volume for comparison
        if self.indicator_config.get("volume"):
            result["vol_sma_20"] = ind.sma(result["volume"], 20)
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

        # Buy conditions: oversold RSI, bullish patterns, golden cross, higher lows
        buy_conditions = []
        if "rsi" in signals.columns:
            buy_conditions.append(signals["rsi"] < rsi_oversold)
        # Patterns: any bullish pattern flagged as True
        bullish_cols = [col for col in patterns.columns if any(keyword in col for keyword in ["bullish", "gap_up", "hammer", "double_bottom", "triple_bottom", "cup_and_handle", "higher_lows"])]
        if bullish_cols:
            buy_conditions.append(patterns[bullish_cols].any(axis=1))
        # Golden cross indicator
        if "golden_cross" in signals.columns:
            buy_conditions.append(signals["golden_cross"])
        # Combine buy conditions (AND across groups)
        if buy_conditions:
            buy_signal = buy_conditions[0]
            for cond in buy_conditions[1:]:
                buy_signal &= cond
            signal = signal.where(~buy_signal, 1)

        # Sell conditions: overbought RSI, bearish patterns, death cross, lower highs
        sell_conditions = []
        if "rsi" in signals.columns:
            sell_conditions.append(signals["rsi"] > rsi_overbought)
        bearish_cols = [col for col in patterns.columns if any(keyword in col for keyword in ["bearish", "gap_down", "shooting_star", "double_top", "triple_top", "rising_wedge", "lower_highs"])]
        if bearish_cols:
            sell_conditions.append(patterns[bearish_cols].any(axis=1))
        if "death_cross" in signals.columns:
            sell_conditions.append(signals["death_cross"])
        if sell_conditions:
            sell_signal = sell_conditions[0]
            for cond in sell_conditions[1:]:
                sell_signal &= cond
            signal = signal.where(~sell_signal, -1)

        signals["signal"] = signal
        return signals
