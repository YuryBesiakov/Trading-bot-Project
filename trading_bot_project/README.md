# Automated Stock Trading Bot

This repository contains a Python‑based trading bot designed for paper trading on the **Alpaca** brokerage platform.  The bot fetches market data, applies a variety of technical analysis patterns and indicators, and can be configured to place paper trades when signals are triggered.  It also includes tooling for backtesting strategies over historical data, scheduling periodic analysis, and sending notifications via Telegram, Slack or email.

## Features

* **Data fetching**: Retrieve historical and real‑time price data using Alpaca’s API (configured for paper trading).  The polling interval (hourly, daily, etc.) is configurable.
* **Technical analysis**: Implements a suite of commonly used indicators and patterns, including:
  * Bullish and bearish engulfing candles
  * Gap detection (up and down gaps)
  * Hammer and shooting star formations
  * Double/triple tops and bottoms
  * Cup and handle pattern
  * Rising wedge pattern
  * Moving averages (simple and exponential) for 50, 100 and 200 periods
  * Golden cross and death cross detection
  * Higher‑lows and lower‑highs detection
  * Volume analysis and relative strength index (RSI)
  * Support/resistance and Fibonacci retracement levels (basic implementation)
* **Backtesting**: Evaluate the performance of strategies over historical data.  Reports key metrics such as cumulative return, maximum drawdown and win/loss ratio.
* **Live/paper trading**: Generate trading signals based on strategy output and place orders on Alpaca’s paper trading account.  All trades and actions are logged for transparency and debugging.
* **Scheduling**: Run the strategy at a configurable interval using a simple scheduler.  The default is hourly.
* **Notifications**: Send alerts for executed trades or errors via Telegram, Slack or email.  Notification channels are optional and configured in the project’s YAML configuration file.
* **Extensible design**: New patterns, indicators or machine learning models can be added by implementing additional functions in the respective modules and registering them in `strategy.py`.

## Project structure

```
trading_bot_project/
├── Dockerfile             # Container definition for deployment
├── README.md              # Project overview and instructions
├── requirements.txt       # Python dependencies
├── config_example.yaml    # Example configuration (copy and customise)
├── main.py                # Entry point for running the bot
├── backtester.py          # Backtesting engine
├── data_fetcher.py        # Historical and real‑time data access
├── indicators.py          # Technical indicators (MA, RSI, Fibonacci, etc.)
├── patterns.py            # Candlestick and chart pattern detection
├── strategy.py            # Combines indicators and patterns into trading signals
├── trade_executor.py      # Handles paper trading with Alpaca
├── notifier.py            # Telegram, Slack and email notifications
├── scheduler.py           # Task scheduling
├── utils.py               # Utility functions (logging, config loading)
└── __init__.py            # Marks this directory as a package
```

## Quick start

1. **Clone the repository** and install the dependencies.  It’s recommended to use a virtual environment.

```bash
git clone https://github.com/YuryBesiakov/Trading-bot-Project.git
cd Trading-bot-Project/trading_bot_project
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. **Create a configuration file** by copying `config_example.yaml` to `config.yaml` and editing the values for your Alpaca API keys, symbols to trade, schedule, position sizing and notification preferences.

3. **Run a backtest** on your strategy over a date range to verify its behaviour:

```bash
python backtester.py --config config.yaml --start 2023-01-01 --end 2024-01-01
```

4. **Start the trading bot** for live paper trading (this will run indefinitely and execute on the configured schedule):

```bash
python main.py --config config.yaml
```

5. **Build and run with Docker** (optional).  This uses the provided `Dockerfile` for a reproducible environment.

```bash
docker build -t trading-bot .
docker run -e ALPACA_KEY_ID=your_key -e ALPACA_SECRET_KEY=your_secret \
           -v $(pwd)/config.yaml:/app/config.yaml trading-bot
```

## Configuration

The bot is configured via a YAML file.  An example is provided in `config_example.yaml`.  Important parameters include:

* **alpaca**: API credentials and endpoint (e.g. paper trading URL).
* **symbols**: List of tickers to analyse and trade.
* **interval**: Data frequency (e.g. `1H` for hourly, `1D` for daily).  Accepts any pandas offset alias.
* **strategy**: Enable/disable specific patterns and indicators and specify thresholds.
* **backtest**: Start and end dates and initial capital for backtesting.
* **trade**: Position sizing and risk management settings.
* **notifications**: Configure Telegram, Slack or email settings for alerts.

See the sample configuration file for a complete list of fields and their descriptions.

## Adding new indicators or patterns

The codebase is organised to make it easy to extend.  To add a new technical indicator or pattern:

1. Implement the calculation or detection logic in either `indicators.py` or `patterns.py`.
2. Expose the function by adding it to the relevant registry in `strategy.py` so that it can be enabled via the configuration.
3. Update the documentation (here and in the configuration example) to describe the new option and its parameters.

## License

This project is provided for educational purposes.  Use at your own risk.  No warranty or guarantee of performance is provided.
.