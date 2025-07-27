"""Candlestick and chart pattern detection.

This module implements basic detection for a variety of candlestick
patterns and chart formations.  Most functions return a boolean
Series aligned with the input DataFrame’s index, where ``True``
indicates the presence of the pattern on that bar.  Some complex
formations (e.g. cup and handle, rising wedge) are provided as stubs
that return ``False`` for all rows – these can be enhanced later.
"""

from __future__ import annotations

import pandas as pd
import numpy as np


def detect_bullish_engulfing(df: pd.DataFrame) -> pd.Series:
    """Identify bullish engulfing candles.

    A bullish engulfing pattern occurs when a small bearish candle is
    followed by a larger bullish candle whose body completely
    engulfs the previous day’s body.

    Args:
        df: DataFrame with ``open`` and ``close`` columns.

    Returns:
        Boolean Series marking rows where the pattern occurs.
    """
    prev_open = df["open"].shift(1)
    prev_close = df["close"].shift(1)
    # Previous candle bearish, current bullish
    prev_bearish = prev_close < prev_open
    curr_bullish = df["close"] > df["open"]
    # Current body engulfs previous body
    engulf = (df["open"] <= prev_close) & (df["close"] >= prev_open)
    pattern = prev_bearish & curr_bullish & engulf
    return pattern.fillna(False)


def detect_bearish_engulfing(df: pd.DataFrame) -> pd.Series:
    """Identify bearish engulfing candles.

    A bearish engulfing pattern occurs when a small bullish candle is
    followed by a larger bearish candle whose body engulfs the
    previous day’s body.

    Args:
        df: DataFrame with ``open`` and ``close`` columns.

    Returns:
        Boolean Series marking rows where the pattern occurs.
    """
    prev_open = df["open"].shift(1)
    prev_close = df["close"].shift(1)
    prev_bullish = prev_close > prev_open
    curr_bearish = df["close"] < df["open"]
    engulf = (df["open"] >= prev_close) & (df["close"] <= prev_open)
    pattern = prev_bullish & curr_bearish & engulf
    return pattern.fillna(False)


def detect_gap_up(df: pd.DataFrame) -> pd.Series:
    """Detect up gaps between consecutive bars.

    A gap up occurs when the current bar’s low price is above the
    previous bar’s high price.

    Args:
        df: DataFrame with ``high`` and ``low`` columns.

    Returns:
        Boolean Series marking rows where a gap up occurs.
    """
    prev_high = df["high"].shift(1)
    gap = df["low"] > prev_high
    return gap.fillna(False)


def detect_gap_down(df: pd.DataFrame) -> pd.Series:
    """Detect down gaps between consecutive bars.

    A gap down occurs when the current bar’s high price is below the
    previous bar’s low price.

    Args:
        df: DataFrame with ``high`` and ``low`` columns.

    Returns:
        Boolean Series marking rows where a gap down occurs.
    """
    prev_low = df["low"].shift(1)
    gap = df["high"] < prev_low
    return gap.fillna(False)


def detect_hammer(df: pd.DataFrame) -> pd.Series:
    """Detect hammer candlesticks.

    A hammer has a small real body near the top of the candle with a
    long lower shadow.  This implementation uses simple ratios to
    approximate the pattern.

    Args:
        df: DataFrame with ``open``, ``close``, ``high`` and ``low``.

    Returns:
        Boolean Series indicating hammers.
    """
    body = (df["close"] - df["open"]).abs()
    candle_range = df["high"] - df["low"]
    lower_shadow = df[["open", "close"]].min(axis=1) - df["low"]
    upper_shadow = df["high"] - df[["open", "close"]].max(axis=1)
    # Conditions: small body, long lower shadow, little to no upper shadow
    small_body = body / candle_range < 0.3
    long_lower = lower_shadow / candle_range > 0.5
    tiny_upper = upper_shadow / candle_range < 0.2
    return (small_body & long_lower & tiny_upper).fillna(False)


def detect_shooting_star(df: pd.DataFrame) -> pd.Series:
    """Detect shooting star candlesticks.

    A shooting star has a small body near the bottom of the candle with
    a long upper shadow.  Mirror of the hammer pattern.

    Args:
        df: DataFrame with OHLC data.

    Returns:
        Boolean Series indicating shooting stars.
    """
    body = (df["close"] - df["open"]).abs()
    candle_range = df["high"] - df["low"]
    lower_shadow = df[["open", "close"]].min(axis=1) - df["low"]
    upper_shadow = df["high"] - df[["open", "close"]].max(axis=1)
    small_body = body / candle_range < 0.3
    long_upper = upper_shadow / candle_range > 0.5
    tiny_lower = lower_shadow / candle_range < 0.2
    return (small_body & long_upper & tiny_lower).fillna(False)


def detect_doji(df: pd.DataFrame, tolerance: float = 0.001) -> pd.Series:
    """Detect Doji candlesticks.

    A Doji occurs when the opening and closing prices are virtually equal,
    creating a cross-like appearance. This indicates market indecision.

    Args:
        df: DataFrame with OHLC data.
        tolerance: Maximum relative difference between open and close (default 0.1%).

    Returns:
        Boolean Series indicating Doji candles.
    """
    body = (df["close"] - df["open"]).abs()
    candle_range = df["high"] - df["low"]
    # Avoid division by zero
    candle_range = candle_range.replace(0, pd.NA)
    body_ratio = body / candle_range
    # Doji: very small body relative to range
    is_doji = body_ratio < tolerance
    return is_doji.fillna(False)


def detect_dragonfly_doji(df: pd.DataFrame, tolerance: float = 0.001) -> pd.Series:
    """Detect Dragonfly Doji candlesticks.

    A Dragonfly Doji has a small body at the top with a long lower shadow
    and little to no upper shadow. Often bullish reversal signal.

    Args:
        df: DataFrame with OHLC data.
        tolerance: Body size tolerance relative to total range.

    Returns:
        Boolean Series indicating Dragonfly Doji candles.
    """
    body = (df["close"] - df["open"]).abs()
    candle_range = df["high"] - df["low"]
    lower_shadow = df[["open", "close"]].min(axis=1) - df["low"]
    upper_shadow = df["high"] - df[["open", "close"]].max(axis=1)
    
    # Avoid division by zero
    candle_range = candle_range.replace(0, pd.NA)
    
    small_body = body / candle_range < tolerance
    long_lower = lower_shadow / candle_range > 0.6
    tiny_upper = upper_shadow / candle_range < 0.1
    
    return (small_body & long_lower & tiny_upper).fillna(False)


def detect_gravestone_doji(df: pd.DataFrame, tolerance: float = 0.001) -> pd.Series:
    """Detect Gravestone Doji candlesticks.

    A Gravestone Doji has a small body at the bottom with a long upper shadow
    and little to no lower shadow. Often bearish reversal signal.

    Args:
        df: DataFrame with OHLC data.
        tolerance: Body size tolerance relative to total range.

    Returns:
        Boolean Series indicating Gravestone Doji candles.
    """
    body = (df["close"] - df["open"]).abs()
    candle_range = df["high"] - df["low"]
    lower_shadow = df[["open", "close"]].min(axis=1) - df["low"]
    upper_shadow = df["high"] - df[["open", "close"]].max(axis=1)
    
    # Avoid division by zero
    candle_range = candle_range.replace(0, pd.NA)
    
    small_body = body / candle_range < tolerance
    long_upper = upper_shadow / candle_range > 0.6
    tiny_lower = lower_shadow / candle_range < 0.1
    
    return (small_body & long_upper & tiny_lower).fillna(False)


def detect_spinning_top(df: pd.DataFrame) -> pd.Series:
    """Detect Spinning Top candlesticks.

    A Spinning Top has a small body with long shadows on both sides,
    indicating indecision in the market.

    Args:
        df: DataFrame with OHLC data.

    Returns:
        Boolean Series indicating Spinning Top candles.
    """
    body = (df["close"] - df["open"]).abs()
    candle_range = df["high"] - df["low"]
    lower_shadow = df[["open", "close"]].min(axis=1) - df["low"]
    upper_shadow = df["high"] - df[["open", "close"]].max(axis=1)
    
    # Avoid division by zero
    candle_range = candle_range.replace(0, pd.NA)
    
    small_body = body / candle_range < 0.25
    balanced_shadows = (lower_shadow / candle_range > 0.3) & (upper_shadow / candle_range > 0.3)
    
    return (small_body & balanced_shadows).fillna(False)


def detect_double_top(df: pd.DataFrame, window: int = 5, tolerance: float = 0.005) -> pd.Series:
    """Simplistic double top detection.

    This looks for two local maxima of similar height within a
    ``window`` period.  The tolerance parameter controls how similar
    the peaks must be (as a fraction of the price).  The function
    returns ``True`` on the second peak.

    Args:
        df: DataFrame with a ``high`` column.
        window: Lookback window to search for peaks.
        tolerance: Relative tolerance between peak values.

    Returns:
        Boolean Series where ``True`` indicates a double top.
    """
    high = df["high"]
    pattern = pd.Series(False, index=high.index)
    # We need at least 2*window data points
    if len(high) < window * 2:
        return pattern
    for i in range(window * 2, len(high)):
        segment = high.iloc[i - window * 2 : i]
        first_peak = segment[:window].max()
        second_peak = segment[window:].max()
        if abs(first_peak - second_peak) / first_peak < tolerance:
            # Mark the index of second peak as pattern
            if high.iloc[i] == second_peak:
                pattern.iloc[i] = True
    return pattern


def detect_double_bottom(df: pd.DataFrame, window: int = 5, tolerance: float = 0.005) -> pd.Series:
    """Simplistic double bottom detection.

    Mirror of ``detect_double_top`` but for lows.  Returns ``True`` on
    the second trough when two similar lows occur within a window.
    """
    low = df["low"]
    pattern = pd.Series(False, index=low.index)
    if len(low) < window * 2:
        return pattern
    for i in range(window * 2, len(low)):
        segment = low.iloc[i - window * 2 : i]
        first_trough = segment[:window].min()
        second_trough = segment[window:].min()
        if abs(first_trough - second_trough) / first_trough < tolerance:
            if low.iloc[i] == second_trough:
                pattern.iloc[i] = True
    return pattern


def detect_triple_top(df: pd.DataFrame) -> pd.Series:
    """Placeholder for triple top detection.

    Returns a series of False values since advanced pattern
    recognition is not implemented.  Extend this function with
    appropriate logic if required.
    """
    return pd.Series(False, index=df.index)


def detect_triple_bottom(df: pd.DataFrame) -> pd.Series:
    """Placeholder for triple bottom detection.

    Returns a series of False values.  Extend as needed.
    """
    return pd.Series(False, index=df.index)


def detect_cup_and_handle(df: pd.DataFrame) -> pd.Series:
    """Placeholder for cup and handle pattern detection.

    Returns all False values.  Implement actual detection if required.
    """
    return pd.Series(False, index=df.index)


def detect_rising_wedge(df: pd.DataFrame) -> pd.Series:
    """Placeholder for rising wedge pattern detection.

    Returns all False values.  Implement actual detection if required.
    """
    return pd.Series(False, index=df.index
    )