"""Entry point for running the trading bot.

This script loads the user’s configuration, instantiates the strategy,
scheduler, notifier and trade executor, and orchestrates the periodic
execution of the trading logic.  It is intended to be run with a
configuration file specifying Alpaca credentials, symbols to trade and
strategy settings.
"""

from __future__ import annotations

import argparse
import datetime as dt
from typing import Dict, List

import pandas as pd

from .utils import load_config, setup_logger
from .data_fetcher import get_historical_data
from .strategy import Strategy
from .trade_executor import TradeExecutor
from .notifier import Notifier
from .scheduler import Scheduler


logger = setup_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the automated trading bot")
    parser.add_argument("--config", required=True, help="Path to YAML configuration file")
    return parser.parse_args()


class TradingBot:
    """Encapsulates state and logic for live paper trading."""

    def __init__(self, config: Dict[str, any]):
        self.config = config
        self.alpaca_cfg = config.get("alpaca", {})
        self.symbols: List[str] = config.get("symbols", [])
        self.interval: str = config.get("interval", "1H")
        self.strategy_cfg: Dict[str, any] = config.get("strategy", {})
        self.strategy_cfg["interval"] = self.interval
        self.trade_cfg: Dict[str, any] = config.get("trade", {})
        self.notifier = Notifier(config.get("notifications", {}))
        self.executor = TradeExecutor(self.alpaca_cfg)
        self.positions: Dict[str, int] = {s: 0 for s in self.symbols}

    def run_analysis(self) -> None:
        """Run one cycle of analysis and trading across all symbols."""
        for symbol in self.symbols:
            try:
                logger.info("Running analysis for %s", symbol)
                # Determine lookback period based on interval (e.g. last 300 bars)
                lookback_bars = 300
                now = dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc)
                # Convert interval to approximate timedelta
                interval = self.interval.lower()
                if interval.endswith("h"):
                    period = dt.timedelta(hours=int(interval[:-1]))
                elif interval.endswith("d"):
                    period = dt.timedelta(days=int(interval[:-1]))
                elif interval.endswith("min") or interval.endswith("t"):
                    minutes = int(interval.split("min")[0] if "min" in interval else interval[:-1])
                    period = dt.timedelta(minutes=minutes)
                else:
                    period = dt.timedelta(hours=1)
                start_time = now - lookback_bars * period
                # Fetch historical data
                df = get_historical_data(symbol, start_time, now, self.interval, self.alpaca_cfg)
                if df.empty:
                    logger.warning("No data for %s; skipping", symbol)
                    continue
                strat = Strategy(self.strategy_cfg)
                signals = strat.generate_signals(df)
                if signals.empty:
                    continue
                last_signal = signals.iloc[-1]["signal"]
                position = self.positions.get(symbol, 0)
                # Determine action
                if last_signal == 1 and position <= 0:
                    # Close short if needed
                    if position < 0:
                        closed = self.executor.close_position(symbol)
                        if closed:
                            self.notifier.notify(
                                f"Closed short {symbol}", f"Closed short position on {symbol}"
                            )
                            self.positions[symbol] = 0
                    # Open long
                    qty = 1  # quantity can be derived from cash; simplified to 1 share
                    order_id = self.executor.submit_order(symbol, qty, "buy")
                    if order_id:
                        self.notifier.notify(
                            f"Bought {symbol}",
                            f"Opened long position of {qty} shares on {symbol} at signal time",
                        )
                        self.positions[symbol] = 1
                elif last_signal == -1 and position >= 0:
                    # Close long if needed
                    if position > 0:
                        closed = self.executor.close_position(symbol)
                        if closed:
                            self.notifier.notify(
                                f"Closed long {symbol}", f"Closed long position on {symbol}"
                            )
                            self.positions[symbol] = 0
                    # Open short
                    qty = 1
                    order_id = self.executor.submit_order(symbol, qty, "sell")
                    if order_id:
                        self.notifier.notify(
                            f"Shorted {symbol}",
                            f"Opened short position of {qty} shares on {symbol} at signal time",
                        )
                        self.positions[symbol] = -1
                else:
                    logger.info("No action for %s (signal=%s, position=%s)", symbol, last_signal, position)
            except Exception as e:
                logger.exception("Error running analysis for %s: %s", symbol, e)
                self.notifier.notify(
                    f"Error for {symbol}", f"An error occurred while analysing {symbol}: {e}"
                )

    def start(self):
        """Start the periodic scheduler."""
        sched_cfg = self.config.get("schedule", {})
        interval = sched_cfg.get("interval", self.interval)
        scheduler = Scheduler(interval, self.run_analysis)
        scheduler.start()
        try:
            # Keep the main thread alive
            while True:
                pass
        except KeyboardInterrupt:
            scheduler.stop()
            logger.info("Bot stopped by user")


def main():
    args = parse_args()
    config = load_config(args.config)
    bot = TradingBot(config)
    bot.start()


if __name__ == "__main__":
    main()
