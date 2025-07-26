"""Trading bot package initialisation.

This package contains modules for fetching data, computing technical
indicators and patterns, building strategies, running backtests,
scheduling jobs, executing paper trades on Alpaca and sending
notifications.  See individual modules for more details.
"""

__all__ = [
    "data_fetcher",
    "indicators",
    "patterns",
    "strategy",
    "backtester",
    "notifier",
    "trade_executor",
    "scheduler",
    "utils",
    "main",
]