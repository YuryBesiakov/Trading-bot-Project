"""Historical backtesting for trading strategies.

This module defines a simple backtesting engine that runs a strategy
over historical price data and reports performance metrics.  It
supports long and short positions with a configurable cash allocation
per trade.  The engine is intentionally simplified: it assumes
immediate fills at the next bar’s open price, ignores transaction
costs and slippage, and does not model margin requirements.  It is
intended for educational purposes and as a starting point for more
sophisticated backtesting frameworks.
"""

from __future__ import annotations

import argparse
import datetime as dt
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .data_fetcher import get_historical_data
from .strategy import Strategy
from .utils import load_config, setup_logger


logger = setup_logger(__name__)


@dataclass
class Trade:
    symbol: str
    direction: int  # 1 for long, -1 for short
    entry_time: dt.datetime
    entry_price: float
    exit_time: dt.datetime
    exit_price: float
    quantity: float
    profit: float


class Backtester:
    """Backtesting engine for evaluating trading strategies."""

    def __init__(
        self,
        strategy_config: Dict[str, any],
        trade_config: Dict[str, any],
        alpaca_cfg: Dict[str, str],
    ):
        self.strategy_cfg = strategy_config
        self.trade_cfg = trade_config
        self.alpaca_cfg = alpaca_cfg
        self.cash_allocation = trade_config.get("cash_allocation", 0.1)

    def run(self, symbol: str, start: dt.datetime, end: dt.datetime, initial_capital: float) -> Dict[str, any]:
        """Run a backtest for a single symbol.

        Args:
            symbol: Ticker to backtest.
            start: Start date/time (UTC).
            end: End date/time (UTC).
            initial_capital: Starting capital for the backtest.

        Returns:
            Dictionary with performance metrics and trade history.
        """
        # Fetch historical data
        df = get_historical_data(symbol, start, end, self.strategy_cfg.get("interval", "1D"), self.alpaca_cfg)
        if df.empty:
            logger.warning("No data for symbol %s; skipping", symbol)
            return {}
        # Generate signals
        strat = Strategy(self.strategy_cfg)
        signals = strat.generate_signals(df)
        # Use open price for entry/exit
        open_prices = df["open"]
        capital = initial_capital
        position = 0.0  # number of shares held (positive for long, negative for short)
        entry_price = 0.0
        trades: List[Trade] = []
        for idx in signals.index:
            signal = signals.loc[idx, "signal"]
            price = open_prices.loc[idx]
            if signal == 1 and position == 0:
                # Open long position
                allocation = capital * self.cash_allocation
                qty = allocation / price if price > 0 else 0
                position = qty
                capital -= allocation
                entry_price = price
                entry_time = idx
                logger.debug("Enter long %s at %s qty %.4f price %.2f", symbol, idx, qty, price)
            elif signal == -1 and position == 0:
                # Open short position
                allocation = capital * self.cash_allocation
                qty = allocation / price if price > 0 else 0
                position = -qty
                capital += allocation  # For short we assume we receive cash now
                entry_price = price
                entry_time = idx
                logger.debug("Enter short %s at %s qty %.4f price %.2f", symbol, idx, qty, price)
            elif signal == -1 and position > 0:
                # Close long position
                profit = (price - entry_price) * position
                capital += price * position + profit
                trades.append(
                    Trade(
                        symbol=symbol,
                        direction=1,
                        entry_time=entry_time,
                        entry_price=entry_price,
                        exit_time=idx,
                        exit_price=price,
                        quantity=position,
                        profit=profit,
                    )
                )
                logger.debug("Exit long %s at %s qty %.4f price %.2f profit %.2f", symbol, idx, position, price, profit)
                position = 0
            elif signal == 1 and position < 0:
                # Close short position
                profit = (entry_price - price) * abs(position)
                capital += profit  # Profit added; quantity returns to zero
                trades.append(
                    Trade(
                        symbol=symbol,
                        direction=-1,
                        entry_time=entry_time,
                        entry_price=entry_price,
                        exit_time=idx,
                        exit_price=price,
                        quantity=abs(position),
                        profit=profit,
                    )
                )
                logger.debug("Exit short %s at %s qty %.4f price %.2f profit %.2f", symbol, idx, -position, price, profit)
                position = 0
            # Hold positions until end; profit/loss will be realised at close
        # Close any open position at the end price
        if position != 0:
            final_price = open_prices.iloc[-1]
            if position > 0:
                profit = (final_price - entry_price) * position
                capital += final_price * position + profit
                trades.append(
                    Trade(
                        symbol=symbol,
                        direction=1,
                        entry_time=entry_time,
                        entry_price=entry_price,
                        exit_time=signals.index[-1],
                        exit_price=final_price,
                        quantity=position,
                        profit=profit,
                    )
                )
            else:
                profit = (entry_price - final_price) * abs(position)
                capital += profit
                trades.append(
                    Trade(
                        symbol=symbol,
                        direction=-1,
                        entry_time=entry_time,
                        entry_price=entry_price,
                        exit_time=signals.index[-1],
                        exit_price=final_price,
                        quantity=abs(position),
                        profit=profit,
                    )
                )
            position = 0

        # Compute metrics
        total_profit = sum(t.profit for t in trades)
        net_return = (capital - initial_capital) / initial_capital
        # Win/loss ratio
        wins = len([t for t in trades if t.profit > 0])
        losses = len([t for t in trades if t.profit <= 0])
        win_loss_ratio = wins / losses if losses > 0 else float("inf")
        # Drawdown: compute equity curve from trades
        equity_curve = initial_capital + pd.Series([0] + [sum(t.profit for t in trades[:i+1]) for i in range(len(trades))])
        rolling_max = equity_curve.cummax()
        drawdown = (equity_curve - rolling_max) / rolling_max
        max_drawdown = drawdown.min() if not drawdown.empty else 0.0

        metrics = {
            "symbol": symbol,
            "initial_capital": initial_capital,
            "final_capital": capital,
            "net_return": net_return,
            "total_profit": total_profit,
            "max_drawdown": float(max_drawdown),
            "wins": wins,
            "losses": losses,
            "win_loss_ratio": win_loss_ratio,
            "trades": trades,
        }
        return metrics


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for running the backtester."""
    parser = argparse.ArgumentParser(description="Run backtests for the trading bot")
    parser.add_argument("--config", required=True, help="Path to YAML configuration file")
    parser.add_argument("--symbol", help="Single symbol to backtest (overrides config)")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--initial_capital", type=float, default=10000.0, help="Initial capital for backtesting"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    alpaca_cfg = config.get("alpaca", {})
    strategy_cfg = config.get("strategy", {})
    strategy_cfg["interval"] = config.get("interval", "1D")
    trade_cfg = config.get("trade", {})
    symbols = [args.symbol] if args.symbol else config.get("symbols", [])
    start_date = dt.datetime.fromisoformat(args.start)
    end_date = dt.datetime.fromisoformat(args.end)
    engine = Backtester(strategy_cfg, trade_cfg, alpaca_cfg)
    for symbol in symbols:
        metrics = engine.run(symbol, start_date, end_date, args.initial_capital)
        if not metrics:
            continue
        logger.info(
            "Backtest for %s: return %.2f%%, max drawdown %.2f%%, win/loss %d/%d", 
            symbol,
            metrics["net_return"] * 100,
            metrics["max_drawdown"] * 100,
            metrics["wins"],
            metrics["losses"],
        )


if __name__ == "__main__":
    main()
