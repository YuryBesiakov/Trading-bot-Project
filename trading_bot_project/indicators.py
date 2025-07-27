"""Technical indicator calculations.

This module provides functions to compute common technical indicators
used in trading strategies.  Indicators operate on pandas Series or
DataFrames and return new Series/DataFrames that can be merged with
the original price data.  Where practical the calculations are
implemented directly using pandas to avoid external dependencies.  If
you prefer to use a specialised library (e.g. TA‑Lib or pandas‑ta),
you can wrap those calls here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


def sma(series: pd.Series, window: int) -> pd.Series:
    """Simple moving average (SMA).

    Args:
        series: Price series (e.g. close prices).
        window: Number of periods to average.

    Returns:
        SMA series aligned with the input index.
    """
    return series.rolling(window=window, min_periods=window).mean()


def ema(series: pd.Series, window: int) -> pd.Series:
    """Exponential moving average (EMA).

    Args:
        series: Price series (e.g. close prices).
        window: Smoothing window length.

    Returns:
        EMA series.
    """
    return series.ewm(span=window, adjust=False).mean()


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Relative strength index (RSI).

    Computes the RSI using the classic Wilder’s smoothing method.

    Args:
        series: Price series (typically close prices).
        window: Number of periods (default 14).

    Returns:
        RSI values between 0 and 100.
    """
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - 100 / (1 + rs)
    return rsi


def pivot_points(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate daily pivot point, support and resistance levels.

    Uses the standard formula: P = (H + L + C) / 3.  Support/resistance
    levels are derived using typical Fibonacci ratios.  Only works on
    daily data (one row per day).  For intraday bars you should
    resample to daily first.

    Args:
        df: DataFrame with columns ``high``, ``low`` and ``close``.

    Returns:
        DataFrame with pivot point (p), supports (s1, s2, s3) and
        resistances (r1, r2, r3).
    """
    p = (df["high"] + df["low"] + df["close"]) / 3.0
    r1 = 2 * p - df["low"]
    s1 = 2 * p - df["high"]
    r2 = p + (df["high"] - df["low"])
    s2 = p - (df["high"] - df["low"])
    r3 = df["high"] + 2 * (p - df["low"])
    s3 = df["low"] - 2 * (df["high"] - p)
    return pd.DataFrame({"p": p, "s1": s1, "s2": s2, "s3": s3, "r1": r1, "r2": r2, "r3": r3})


def fibonacci_levels(series: pd.Series) -> Dict[str, float]:
    """Calculate Fibonacci retracement levels for a price series.

    This computes the retracement levels for the highest and lowest
    values in the series.  These levels can be used to identify
    potential support/resistance zones.

    Args:
        series: Series of prices (e.g. closes).

    Returns:
        Dictionary mapping retracement ratios to price levels.
    """
    high = series.max()
    low = series.min()
    diff = high - low
    levels = {
        "0.0%": high,
        "23.6%": high - diff * 0.236,
        "38.2%": high - diff * 0.382,
        "50.0%": high - diff * 0.5,
        "61.8%": high - diff * 0.618,
        "78.6%": high - diff * 0.786,
        "100.0%": low,
    }
    return levels


def golden_cross(short_ma: pd.Series, long_ma: pd.Series) -> pd.Series:
    """Detect golden cross events.

    A golden cross occurs when a short moving average crosses above
    a long moving average.  This function returns a boolean series
    indicating where such crosses occur.

    Args:
        short_ma: Shorter period moving average.
        long_ma: Longer period moving average.

    Returns:
        Series of booleans where ``True`` indicates a golden cross.
    """
    cross_up = (short_ma.shift(1) < long_ma.shift(1)) & (short_ma > long_ma)
    return cross_up


def death_cross(short_ma: pd.Series, long_ma: pd.Series) -> pd.Series:
    """Detect death cross events.

    The inverse of the golden cross: occurs when a short moving
    average crosses below a long moving average.

    Args:
        short_ma: Shorter period moving average.
        long_ma: Longer period moving average.

    Returns:
        Series of booleans where ``True`` indicates a death cross.
    """
    cross_down = (short_ma.shift(1) > long_ma.shift(1)) & (short_ma < long_ma)
    return cross_down


def higher_lows(series: pd.Series, window: int = 3, min_periods: int = 2) -> pd.Series:
    """Identify higher lows in a series.

    This implementation identifies local lows and checks if each new low 
    is higher than the previous significant low, indicating an uptrend.
    Uses a more flexible approach that can detect both strict local minima
    and relative lows within a rolling window.

    Args:
        series: Series of low prices.
        window: Lookback window for local minima detection.
        min_periods: Minimum periods required around a point to consider it a local minimum.

    Returns:
        Boolean series where ``True`` indicates a higher low.
    """
    if len(series) < window * 2:
        return pd.Series(False, index=series.index)
    
    # Method 1: Find local minima using scipy-like approach
    local_lows = pd.Series(False, index=series.index)
    
    # More flexible local minima detection
    for i in range(min_periods, len(series) - min_periods):
        current_val = series.iloc[i]
        
        # Check if current value is lower than values within the window
        left_vals = series.iloc[max(0, i-window):i]
        right_vals = series.iloc[i+1:min(len(series), i+window+1)]
        
        # Current point is a local low if it's <= most surrounding values
        left_condition = len(left_vals) == 0 or current_val <= left_vals.min()
        right_condition = len(right_vals) == 0 or current_val <= right_vals.min()
        
        # Also check that it's not just equal to all values (flat line)
        nearby_vals = pd.concat([left_vals, right_vals])
        has_variation = len(nearby_vals) == 0 or nearby_vals.std() > 0.001
        
        if left_condition and right_condition and has_variation:
            local_lows.iloc[i] = True
    
    # Method 2: Rolling minimum approach as backup
    if local_lows.sum() < 2:
        # Fallback: use rolling minimum with percentage threshold
        rolling_min = series.rolling(window=window, center=True).min()
        threshold = 0.001  # 0.1% threshold for considering a value a "low"
        local_lows = series <= (rolling_min * (1 + threshold))
    
    # Identify higher lows
    higher_lows_signal = pd.Series(False, index=series.index)
    previous_lows = []
    
    for i in range(len(series)):
        if local_lows.iloc[i]:
            current_low = series.iloc[i]
            
            # Check against recent previous lows (not just the immediate previous)
            if previous_lows:
                # Compare against the lowest of recent lows
                recent_min = min(previous_lows[-3:])  # Last 3 lows
                if current_low > recent_min:
                    higher_lows_signal.iloc[i] = True
            
            previous_lows.append(current_low)
            
            # Keep only recent lows to avoid memory issues
            if len(previous_lows) > 10:
                previous_lows = previous_lows[-5:]
    
    return higher_lows_signal


def lower_highs(series: pd.Series, window: int = 3, min_periods: int = 2) -> pd.Series:
    """Identify lower highs in a series.

    This implementation identifies local highs and checks if each new high 
    is lower than the previous significant high, indicating a downtrend.
    Uses a more flexible approach that can detect both strict local maxima
    and relative highs within a rolling window.

    Args:
        series: Series of high prices.
        window: Lookback window for local maxima detection.
        min_periods: Minimum periods required around a point to consider it a local maximum.

    Returns:
        Boolean series where ``True`` indicates a lower high.
    """
    if len(series) < window * 2:
        return pd.Series(False, index=series.index)
    
    # Method 1: Find local maxima using scipy-like approach
    local_highs = pd.Series(False, index=series.index)
    
    # More flexible local maxima detection
    for i in range(min_periods, len(series) - min_periods):
        current_val = series.iloc[i]
        
        # Check if current value is higher than values within the window
        left_vals = series.iloc[max(0, i-window):i]
        right_vals = series.iloc[i+1:min(len(series), i+window+1)]
        
        # Current point is a local high if it's >= most surrounding values
        left_condition = len(left_vals) == 0 or current_val >= left_vals.max()
        right_condition = len(right_vals) == 0 or current_val >= right_vals.max()
        
        # Also check that it's not just equal to all values (flat line)
        nearby_vals = pd.concat([left_vals, right_vals])
        has_variation = len(nearby_vals) == 0 or nearby_vals.std() > 0.001
        
        if left_condition and right_condition and has_variation:
            local_highs.iloc[i] = True
    
    # Method 2: Rolling maximum approach as backup
    if local_highs.sum() < 2:
        # Fallback: use rolling maximum with percentage threshold
        rolling_max = series.rolling(window=window, center=True).max()
        threshold = 0.001  # 0.1% threshold for considering a value a "high"
        local_highs = series >= (rolling_max * (1 - threshold))
    
    # Identify lower highs
    lower_highs_signal = pd.Series(False, index=series.index)
    previous_highs = []
    
    for i in range(len(series)):
        if local_highs.iloc[i]:
            current_high = series.iloc[i]
            
            # Check against recent previous highs (not just the immediate previous)
            if previous_highs:
                # Compare against the highest of recent highs
                recent_max = max(previous_highs[-3:])  # Last 3 highs
                if current_high < recent_max:
                    lower_highs_signal.iloc[i] = True
            
            previous_highs.append(current_high)
            
            # Keep only recent highs to avoid memory issues
            if len(previous_highs) > 10:
                previous_highs = previous_highs[-5:]
    
    return lower_highs_signal


def simple_higher_lows(series: pd.Series, lookback: int = 5) -> pd.Series:
    """Simple alternative implementation for higher lows detection.
    
    Compares current low against the minimum of recent lows in a rolling window.
    More straightforward but potentially less precise than the main implementation.
    
    Args:
        series: Series of low prices.
        lookback: Number of periods to look back for comparison.
        
    Returns:
        Boolean series where True indicates a potential higher low.
    """
    rolling_min = series.rolling(window=lookback).min()
    # Current value is higher than the minimum of recent values
    higher_low_signal = (series > rolling_min.shift(1)) & (series.shift(1) <= rolling_min.shift(2))
    return higher_low_signal.fillna(False)


def simple_lower_highs(series: pd.Series, lookback: int = 5) -> pd.Series:
    """Simple alternative implementation for lower highs detection.
    
    Compares current high against the maximum of recent highs in a rolling window.
    More straightforward but potentially less precise than the main implementation.
    
    Args:
        series: Series of high prices.
        lookback: Number of periods to look back for comparison.
        
    Returns:
        Boolean series where True indicates a potential lower high.
    """
    rolling_max = series.rolling(window=lookback).max()
    # Current value is lower than the maximum of recent values
    lower_high_signal = (series < rolling_max.shift(1)) & (series.shift(1) >= rolling_max.shift(2))
    return lower_high_signal.fillna(False)


def trend_strength(df: pd.DataFrame, short_window: int = 20, long_window: int = 50) -> pd.Series:
    """Calculate trend strength using multiple timeframes.

    Combines price momentum, moving average slopes, and volatility
    to determine trend strength. Returns values between -100 (strong downtrend)
    and +100 (strong uptrend).

    Args:
        df: DataFrame with OHLCV data.
        short_window: Short-term moving average period.
        long_window: Long-term moving average period.

    Returns:
        Series with trend strength values.
    """
    close = df["close"]
    
    # Price momentum (rate of change)
    price_momentum = ((close / close.shift(short_window)) - 1) * 100
    
    # Moving average slopes
    sma_short = sma(close, short_window)
    sma_long = sma(close, long_window)
    
    ma_short_slope = ((sma_short / sma_short.shift(5)) - 1) * 100
    ma_long_slope = ((sma_long / sma_long.shift(10)) - 1) * 100
    
    # Volatility adjustment (lower volatility = stronger trend)
    volatility = close.rolling(window=short_window).std() / close.rolling(window=short_window).mean()
    volatility_factor = 1 - (volatility / volatility.rolling(window=long_window).mean()).clip(0, 2)
    
    # Combine components with weights
    trend_score = (
        price_momentum * 0.4 +
        ma_short_slope * 0.3 +
        ma_long_slope * 0.3
    ) * volatility_factor
    
    # Normalize to -100 to +100 range
    return trend_score.clip(-100, 100)


def weekly_trend_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze trend over weekly timeframes.

    Resamples data to weekly and calculates trend metrics including
    direction, strength, and consistency.

    Args:
        df: DataFrame with OHLCV data and datetime index.

    Returns:
        DataFrame with weekly trend analysis metrics.
    """
    # Resample to weekly data
    weekly = df.resample('W').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    if len(weekly) < 4:
        # Not enough data for weekly analysis
        return pd.DataFrame(index=df.index)
    
    # Weekly price change
    weekly['price_change'] = weekly['close'].pct_change()
    
    # Weekly trend direction (1 = up, -1 = down, 0 = sideways)
    weekly['trend_direction'] = np.where(
        weekly['price_change'] > 0.02, 1,
        np.where(weekly['price_change'] < -0.02, -1, 0)
    )
    
    # Trend consistency (how many of last 4 weeks were in same direction)
    weekly['trend_consistency'] = weekly['trend_direction'].rolling(4).apply(
        lambda x: (x == x.iloc[-1]).sum() / len(x) if len(x) > 0 else 0
    )
    
    # Weekly volatility
    weekly['volatility'] = weekly['close'].rolling(4).std() / weekly['close'].rolling(4).mean()
    
    # Forward fill to daily index
    result = weekly[['trend_direction', 'trend_consistency', 'volatility']].reindex(
        df.index, method='ffill'
    )
    
    return result


def monthly_trend_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze trend over monthly timeframes.

    Similar to weekly analysis but over monthly periods for longer-term
    trend identification.

    Args:
        df: DataFrame with OHLCV data and datetime index.

    Returns:
        DataFrame with monthly trend analysis metrics.
    """
    # Resample to monthly data
    monthly = df.resample('ME').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    if len(monthly) < 3:
        # Not enough data for monthly analysis
        return pd.DataFrame(index=df.index)
    
    # Monthly price change
    monthly['price_change'] = monthly['close'].pct_change()
    
    # Monthly trend direction
    monthly['trend_direction'] = np.where(
        monthly['price_change'] > 0.05, 1,
        np.where(monthly['price_change'] < -0.05, -1, 0)
    )
    
    # Trend momentum (acceleration/deceleration)
    monthly['trend_momentum'] = monthly['price_change'].diff()
    
    # Monthly support/resistance levels
    monthly['monthly_high'] = monthly['high'].rolling(3).max()
    monthly['monthly_low'] = monthly['low'].rolling(3).min()
    
    # Forward fill to daily index
    result = monthly[['trend_direction', 'trend_momentum', 'monthly_high', 'monthly_low']].reindex(
        df.index, method='ffill'
    )
    
    return result


def volume_trend_analysis(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Enhanced volume analysis relative to price trends.

    Analyzes volume patterns to confirm or diverge from price movements.
    
    Args:
        df: DataFrame with OHLCV data.
        window: Period for volume moving average.

    Returns:
        DataFrame with volume analysis metrics.
    """
    close = df["close"]
    volume = df["volume"]
    
    # Volume moving average
    vol_ma = sma(volume, window)
    
    # Volume ratio (current vs average)
    vol_ratio = volume / vol_ma
    
    # Price change
    price_change = close.pct_change()
    
    # Volume-Price Trend (VPT)
    vpt = (volume * price_change).cumsum()
    
    # On-Balance Volume (OBV)
    obv = (volume * np.where(price_change > 0, 1, 
                            np.where(price_change < 0, -1, 0))).cumsum()
    
    # Volume trend (rising/falling)
    vol_trend = np.where(
        vol_ratio > 1.2, 1,  # Rising volume
        np.where(vol_ratio < 0.8, -1, 0)  # Falling volume
    )
    
    # Volume-price confirmation
    # 1 = volume confirms price, -1 = volume diverges, 0 = neutral
    vol_price_confirmation = np.where(
        (price_change > 0) & (vol_trend == 1), 1,  # Up price + up volume
        np.where((price_change < 0) & (vol_trend == 1), 1,  # Down price + up volume
                np.where((price_change > 0) & (vol_trend == -1), -1,  # Up price + down volume
                        np.where((price_change < 0) & (vol_trend == -1), -1, 0)))  # Down price + down volume
    )
    
    return pd.DataFrame({
        'vol_ratio': vol_ratio,
        'vol_trend': vol_trend,
        'vpt': vpt,
        'obv': obv,
        'vol_price_confirmation': vol_price_confirmation
    }, index=df.index)


def advanced_support_resistance(df: pd.DataFrame, window: int = 20, min_touches: int = 2) -> pd.DataFrame:
    """Advanced support and resistance level detection.

    Identifies dynamic support/resistance levels based on price action
    and volume confirmation.

    Args:
        df: DataFrame with OHLCV data.
        window: Lookback window for level detection.
        min_touches: Minimum number of touches to confirm a level.

    Returns:
        DataFrame with support/resistance levels and strength.
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]
    volume = df["volume"]
    
    # Rolling highs and lows
    resistance_level = high.rolling(window).max()
    support_level = low.rolling(window).min()
    
    # Level strength based on number of touches and volume
    resistance_touches = pd.Series(0, index=df.index)
    support_touches = pd.Series(0, index=df.index)
    
    for i in range(window, len(df)):
        # Check resistance touches
        recent_highs = high.iloc[i-window:i]
        max_high = recent_highs.max()
        touches = (recent_highs >= max_high * 0.995).sum()  # Within 0.5% of max
        resistance_touches.iloc[i] = touches
        
        # Check support touches
        recent_lows = low.iloc[i-window:i]
        min_low = recent_lows.min()
        touches = (recent_lows <= min_low * 1.005).sum()  # Within 0.5% of min
        support_touches.iloc[i] = touches
    
    # Level strength (0-5 scale)
    resistance_strength = np.clip(resistance_touches / min_touches, 0, 5)
    support_strength = np.clip(support_touches / min_touches, 0, 5)
    
    # Distance from current price to levels
    resistance_distance = (resistance_level - close) / close
    support_distance = (close - support_level) / close
    
    return pd.DataFrame({
        'resistance_level': resistance_level,
        'support_level': support_level,
        'resistance_strength': resistance_strength,
        'support_strength': support_strength,
        'resistance_distance': resistance_distance,
        'support_distance': support_distance
    }, index=df.index)