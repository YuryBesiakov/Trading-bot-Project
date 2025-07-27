# Trading Bot Tests

This folder contains various backtesting and validation scripts for the trading bot strategy.

## Test Scripts

### 1. `alpaca_sp500_backtest.py`
- **Purpose**: Full S&P 500 backtest using real Alpaca API data
- **Data Source**: Alpaca Markets API (historical data)
- **Configuration**: Reads from `../trading_bot_project/config.yaml`
- **Features**: 
  - Tests 20+ S&P 500 companies
  - Real market data from Alpaca
  - Comprehensive performance metrics
  - Signal strength analysis

**Usage:**
```bash
cd tests
python alpaca_sp500_backtest.py
```

### 2. `strategy_logic_test.py`  
- **Purpose**: Simulated strategy logic validation
- **Data Source**: Generated synthetic market data
- **Features**:
  - Tests all indicators and patterns
  - Validates signal generation logic
  - Quick execution (no API dependencies)
  - Comprehensive requirements coverage

**Usage:**
```bash
cd tests
python strategy_logic_test.py
```

### 3. `test_enhanced_features.py`
- **Purpose**: Unit tests for enhanced trading features
- **Focus**: Doji patterns, trend analysis, and new indicators
- **Features**:
  - Tests individual pattern detection
  - Validates indicator calculations
  - Sample data generation for testing

**Usage:**
```bash
cd tests
python test_enhanced_features.py
```

### 4. `test_trend_logic.py`
- **Purpose**: Specific tests for higher_lows and lower_highs logic
- **Focus**: Trend detection accuracy
- **Features**:
  - Synthetic trend pattern testing
  - Visual validation (matplotlib plots)
  - Logic verification for trend analysis

**Usage:**
```bash
cd tests
python test_trend_logic.py
```

## Configuration

All tests use the main configuration file:
- `../trading_bot_project/config.yaml`

This ensures consistency between testing and production environments.

## Results

Test results are saved as CSV files in the tests directory:
- `alpaca_sp500_backtest_results.csv` - Alpaca backtest results
- `strategy_logic_test_results.csv` - Simulated test results

## Requirements

Tests require the same dependencies as the main trading bot:
- pandas, numpy, yaml, requests
- yfinance (for Yahoo Finance tests, if any)
- All trading_bot_project modules
