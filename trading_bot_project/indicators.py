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


def higher_lows(series: pd.Series, window: int = 3) -> pd.Series:
    """Identify higher lows in a series.

    This simplistic implementation flags points where the low value is
    higher than the previous ``window`` lows.  Use this as a crude
    trend continuation signal.

    Args:
        series: Series of low prices.
        window: Lookback window for comparison.

    Returns:
        Boolean series where ``True`` indicates a higher low.
    """
    rolling_min = series.rolling(window=window + 1).apply(lambda x: x[-1] > x[:-1].max(), raw=True)
    # Replace NaN with False for initial periods
    return rolling_min.fillna(False).astype(bool)


def lower_highs(series: pd.Series, window: int = 3) -> pd.Series:
    """Identify lower highs in a series.

    Flags points where the high value is lower than the previous
    ``window`` highs.  This is the mirror of ``higher_lows`` for
    downtrend continuation.

    Args:
        series: Series of high prices.
        window: Lookback window for comparison.

    Returns:
        Boolean series where ``True`` indicates a lower high.
    """
    rolling_max = series.rolling(window=window + 1).apply(lambda x: x[-1] < x[:-1].min(), raw=True)
    return rolling_max.fillna(False).astype(bool)