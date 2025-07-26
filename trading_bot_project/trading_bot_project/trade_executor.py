"""Order execution via Alpaca API.

This module defines a simple wrapper around Alpaca’s trading API for
submitting orders and closing positions.  All trading here is done on
the paper trading endpoint specified in the configuration.  For live
trading you must change the API base URL and ensure your account is
authorised for live trading.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import requests

from .utils import setup_logger


logger = setup_logger(__name__)


class TradeExecutor:
    """Execute trades on Alpaca’s paper trading account."""

    def __init__(self, alpaca_cfg: Dict[str, str]):
        self.alpaca_cfg = alpaca_cfg
        self.base_url = alpaca_cfg.get("base_url", "https://paper-api.alpaca.markets")

    def _headers(self) -> Dict[str, str]:
        return {
            "APCA-API-KEY-ID": self.alpaca_cfg["key_id"],
            "APCA-API-SECRET-KEY": self.alpaca_cfg["secret_key"],
            "Content-Type": "application/json",
        }

    def submit_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        type: str = "market",
        time_in_force: str = "gtc",
    ) -> Optional[str]:
        """Submit an order to Alpaca.

        Args:
            symbol: Ticker symbol.
            qty: Number of shares to buy or sell.
            side: ``"buy"`` or ``"sell"``.
            type: Order type (default ``"market"``).
            time_in_force: Order duration (default ``"gtc"``).

        Returns:
            Order ID if successful, ``None`` otherwise.
        """
        endpoint = f"{self.base_url}/v2/orders"
        payload = {
            "symbol": symbol,
            "qty": qty,
            "side": side,
            "type": type,
            "time_in_force": time_in_force,
        }
        try:
            resp = requests.post(endpoint, json=payload, headers=self._headers())
            if resp.status_code in (200, 201):
                order_id = resp.json().get("id")
                logger.info("Submitted %s order for %s qty %.2f (order id: %s)", side, symbol, qty, order_id)
                return order_id
            else:
                logger.error("Failed to submit order: %s", resp.text)
        except Exception as e:
            logger.exception("Error submitting order: %s", e)
        return None

    def close_position(self, symbol: str) -> bool:
        """Close an open position for a symbol.

        Args:
            symbol: Ticker symbol.

        Returns:
            ``True`` if the request succeeded, ``False`` otherwise.
        """
        endpoint = f"{self.base_url}/v2/positions/{symbol}"
        try:
            resp = requests.delete(endpoint, headers=self._headers())
            if resp.status_code == 200:
                logger.info("Closed position for %s", symbol)
                return True
            else:
                logger.error("Failed to close position: %s", resp.text)
        except Exception as e:
            logger.exception("Error closing position: %s", e)
        return False
