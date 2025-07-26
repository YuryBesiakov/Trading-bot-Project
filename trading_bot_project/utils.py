"""Utility functions for the trading bot.

This module provides helpers for configuration loading, logging and
common data operations.  Keeping these utilities separated from the
core trading logic makes it easier to unit test and reuse code across
different scripts (main bot, backtester, etc.).
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(path: str) -> Dict[str, Any]:
    """Load the YAML configuration file.

    Environment variables prefixed with ``ALPACA_`` or other top‑level
    section names will override the corresponding keys in the file.

    Args:
        path: Path to the YAML file on disk.

    Returns:
        Nested dictionary with configuration values.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file {path} does not exist")
    with config_path.open("r") as f:
        config = yaml.safe_load(f)

    # Environment variable overrides.  Flatten keys by section to
    # simplify mapping.  For example, ``ALPACA_KEY_ID`` overrides
    # ``alpaca.key_id`` in the YAML.
    for section, values in config.items():
        if isinstance(values, dict):
            prefix = section.upper() + "_"
            for key in values.keys():
                env_key = prefix + key.upper()
                if env_key in os.environ and os.environ[env_key]:
                    values[key] = os.environ[env_key]

    return config


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Configure and return a logger.

    This helper sets up a consistent logging format across modules.  Use
    different logger names for each module to control verbosity.

    Args:
        name: The logger’s name.  Should typically be ``__name__``.
        level: Logging level (e.g. ``logging.INFO``).

    Returns:
        Configured Logger instance.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        # Avoid adding multiple handlers if logger is already set up
        return logger
    logger.setLevel(level)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger