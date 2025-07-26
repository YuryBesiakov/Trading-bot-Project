"""Market data access for the trading bot.

This module encapsulates all interactions with external data providers
such as Alpaca.  Currently it uses the Alpaca REST API to fetch
historical bar data at a specified resolution.  If you wish to
integrate another provider (e.g. Yahoo Finance, Quandl), add a new
function here and ensure your strategy references the appropriate
provider when retrieving price data.
"""

from __future__ import annotations

import datetime as dt
from typing import Dict, List, Optional

import pandas as pd
import requests

from .utils import setup_logger


logger = setup_logger(__name__)


def _alpaca_headers(alpaca_cfg: Dict[str, str]) -> Dict[str, str]:
    """Construct the headers required by the Alpaca API."""
    return {
        "APCA-API-KEY-ID": alpaca_cfg["key_id"],
        "APCA-API-SECRET-KEY": alpaca_cfg["secret_key"],
    }


def get_historical_data(
    symbol: str,
    start: dt.datetime,
    end: dt.datetime,
    interval: str,
    alpaca_cfg: Dict[str, str],
    limit: int = 1000,
) -> pd.DataFrame:
    """Fetch historical OHLCV bars for a symbol from Alpaca.

    Args:
        symbol: Ticker symbol to fetch (uppercase).
        start: Start date/time (naive datetimes treated as UTC).
        end: End date/time (naive datetimes treated as UTC).
        interval: Bar size (e.g. ``"1H"`` for hourly, ``"1D"`` for daily).  See
            Alpaca documentation for valid values.
        alpaca_cfg: Configuration dictionary containing ``key_id``,
            ``secret_key`` and ``base_url``.
        limit: Maximum number of bars per request.  Alpaca imposes
            limits on the number of bars returned in a single call.  If
            the date range spans more bars than ``limit`` allows the
            function will make multiple calls and concatenate the
            results.

    Returns:
        DataFrame with columns ``timestamp``, ``open``, ``high``, ``low``,
        ``close`` and ``volume``, indexed by timestamp.
    """
    # Ensure datetimes are timezone aware (UTC).  Alpaca expects
    # ISO8601 strings.  If naive, assume UTC.
    start = start if start.tzinfo else start.replace(tzinfo=dt.timezone.utc)
    end = end if end.tzinfo else end.replace(tzinfo=dt.timezone.utc)
    iso_start = start.isoformat()
    iso_end = end.isoformat()

    base_url = alpaca_cfg.get("base_url", "https://data.alpaca.markets")
    endpoint = f"{base_url}/v2/stocks/{symbol}/bars"
    params = {
        "start": iso_start,
        "end": iso_end,
        "timeframe": interval,
        "limit": limit,
    }
    all_bars: List[Dict[str, any]] = []
    # Loop because Alpaca may return partial data if there are more than
    # ``limit`` bars in the range.  The API returns bars sorted by time
    # ascending; after each request move the start time forward to
    # continue fetching.
    current_start = iso_start
    while True:
        params["start"] = current_start
        logger.debug(
            "Requesting Alpaca bars for %s from %s to %s", symbol, current_start, iso_end
        )
        resp = requests.get(endpoint, headers=_alpaca_headers(alpaca_cfg), params=params)
        if resp.status_code != 200:
            raise RuntimeError(
                f"Alpaca API request failed: {resp.status_code} {resp.text}"
            )
        data = resp.json().get("bars", [])
        if not data:
            break
        all_bars.extend(data)
        # If we received fewer than ``limit`` bars, we're done
        if len(data) < limit:
            break
        # Otherwise, advance start to the last returned bar's time plus a
        # millisecond to avoid duplicates.
        last_ts = data[-1]["t"]
        last_dt = pd.to_datetime(last_ts, utc=True)
        current_start = (last_dt + pd.Timedelta(milliseconds=1)).isoformat()
        # Break if we've exceeded the end
        if last_dt >= end:
            break

    if not all_bars:
        logger.warning("No bar data returned for %s", symbol)
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    df = pd.DataFrame(all_bars)
    # Alpaca returns timestamps in ISO format; convert to pandas
    # datetime and set index
    df["timestamp"] = pd.to_datetime(df["t"], utc=True)
    df = df.set_index("timestamp")
    df = df.rename(
        columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}
    )
    df = df[["open", "high", "low", "close", "volume"]]
    return df


def get_latest_bar(
    symbol: str, interval: str, alpaca_cfg: Dict[str, str]
) -> Optional[pd.Series]:
    """Fetch the most recent bar for a symbol.

    This is useful during live trading to obtain the latest candle
    without retrieving an entire historical series.  It simply calls
    ``get_historical_data`` with a narrow window ending at now.

    Args:
        symbol: Ticker symbol.
        interval: Bar size.
        alpaca_cfg: Alpaca configuration.

    Returns:
        A pandas Series representing the latest bar (open, high, low,
        close, volume) indexed by column names, or ``None`` if no data
        is available.
    """
    end_time = dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc)
    # Request a small window before the end to ensure at least one bar
    start_time = end_time - dt.timedelta(days=2)
    df = get_historical_data(symbol, start_time, end_time, interval, alpaca_cfg, limit=1)
    if df.empty:
        return None
    return df.iloc[-1]