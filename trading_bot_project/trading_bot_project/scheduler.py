"""Task scheduling for periodic strategy execution.

This module wraps the ``schedule`` library to run the trading bot at
regular intervals defined in the configuration.  It translates
pandas‑style interval strings (e.g. ``"1H"`` or ``"30min"``) into
appropriate schedule calls.
"""

from __future__ import annotations

import threading
import time
from typing import Callable

import schedule

from .utils import setup_logger


logger = setup_logger(__name__)


def _parse_interval(interval: str) -> schedule.Job:
    """Convert a pandas offset alias into a schedule job object.

    Supports hours (e.g. ``"1H"``), minutes (``"30min"`` or ``"30T"``)
    and days (``"1D"``).  For unsupported formats it defaults to
    hourly.

    Args:
        interval: Interval string from configuration.

    Returns:
        Partially configured schedule job (without action attached).
    """
    interval = interval.lower()
    if interval.endswith("h"):
        hours = int(interval[:-1])
        return schedule.every(hours).hours
    if interval.endswith("d"):
        days = int(interval[:-1])
        return schedule.every(days).days
    if interval.endswith("min") or interval.endswith("t"):
        minutes = int(interval.split("min")[0] if "min" in interval else interval[:-1])
        return schedule.every(minutes).minutes
    # default to 1 hour
    return schedule.every().hour


class Scheduler:
    """Run a job function at a specified interval in a background thread."""

    def __init__(self, interval: str, job_func: Callable[[], None]):
        self.interval = interval
        self.job_func = job_func
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.running = False

    def start(self):
        """Schedule the job and start the background loop."""
        job = _parse_interval(self.interval).do(self.job_func)
        logger.info("Scheduled job every %s", self.interval)
        self.running = True
        self.thread.start()

    def _run_loop(self):
        while self.running:
            schedule.run_pending()
            time.sleep(1)

    def stop(self):
        """Stop the scheduler."""
        self.running = False
        logger.info("Scheduler stopped")
