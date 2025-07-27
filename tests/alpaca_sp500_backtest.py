"""
S&P 500 Backtesting Script using Alpaca API

This script uses your existing Alpaca credentials to test the enhanced 
trading strategy on S&P 500 companies with real market data.
"""

import pandas as pd
import numpy as np
import yaml
import requests
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import our trading modules
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trading_bot_project.strategy import Strategy


# Top 100 S&P 500 companies by market cap
SP500_TOP_100 = [
    'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'NVDA', 'TSLA', 'META', 'BRK/B', 'UNH',
    'JNJ', 'XOM', 'JPM', 'V', 'PG', 'HD', 'CVX', 'MA', 'BAC', 'ABBV',
    'PFE', 'AVGO', 'COST', 'DIS', 'KO', 'MRK', 'WMT', 'PEP', 'TMO', 'RHHBY',
    'ACN', 'LIN', 'ABT', 'CRM', 'VZ', 'ADBE', 'DHR', 'MCD', 'TXN', 'RTX',
    'NKE', 'NEE', 'WFC', 'QCOM', 'UPS', 'T', 'SPGI', 'LOW', 'COP', 'ORCL',
    'IBM', 'AMGN', 'HON', 'UNP', 'ELV', 'AXP', 'LRCX', 'SYK', 'BLK', 'BKNG',
    'DE', 'ADP', 'GE', 'TJX', 'MDT', 'C', 'GILD', 'CVS', 'VRTX', 'REGN',
    'SCHW', 'ETN', 'BSX', 'CAT', 'ISRG', 'MMC', 'ZTS', 'PLD', 'AON', 'EQIX',
    'ITW', 'APD', 'CME', 'GD', 'SHW', 'ICE', 'USB', 'DUK', 'CL', 'SO',
    'EMR', 'NSC', 'TGT', 'MCO', 'FDX', 'CSX', 'WELL', 'PNC', 'COF', 'WM'
]


class AlpacaBacktester:
    """Backtesting engine using Alpaca API for data."""
    
    def __init__(self, config_file_path):
        # Load configuration
        with open(config_file_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        # Alpaca credentials with enhanced headers
        self.alpaca_config = self.config['alpaca']
        self.base_url = self.alpaca_config['data_url']
        self.headers = {
            'APCA-API-KEY-ID': self.alpaca_config['key_id'],
            'APCA-API-SECRET-KEY': self.alpaca_config['secret_key'],
            'User-Agent': 'TradingBot/1.0',
            'Accept': 'application/json'
        }
        
        # Print debug info
        print(f"🔑 Using API Key: {self.alpaca_config['key_id'][:8]}...")
        print(f"🌐 Data URL: {self.base_url}")
        print(f"📝 Paper trading: {self.alpaca_config.get('paper', True)}")
        print("=")
        
        # Load strategy configuration from config file
        if 'strategy' in self.config:
            self.strategy_config = self.config['strategy']
            print(f"✅ Strategy config loaded from config file:")
            print(f"   Patterns enabled: {len([k for k, v in self.strategy_config.get('patterns', {}).items() if v])}")
            print(f"   Indicators enabled: {len([k for k, v in self.strategy_config.get('indicators', {}).items() if v])}")
            print(f"   Thresholds defined: {len(self.strategy_config.get('thresholds', {}))}")
        else:
            print(f"⚠️  No strategy config found in config file, using defaults")
            # Fallback to enhanced default configuration if not in config
            self.strategy_config = {
                "patterns": {
                    "bullish_engulfing": True,
                    "bearish_engulfing": True,
                    "hammer": True,
                    "shooting_star": True,
                    "doji": True,
                    "dragonfly_doji": True,
                    "gravestone_doji": True,
                    "double_bottom": True,
                    "double_top": True,
                    "higher_lows": True,
                    "lower_highs": True,
                },
                "indicators": {
                    "rsi": True,
                    "sma_20": True,
                    "sma_50": True,
                    "sma_200": True,
                    "ema_50": True,
                    "ema_200": True,
                    "trend_strength": True,
                    "weekly_trend": True,
                    "volume_analysis": True,
                    "golden_cross": True,
                    "death_cross": True,
                },
                "thresholds": {
                    "rsi_oversold": 30,
                    "rsi_overbought": 70,
                    "trend_strength_buy": 15,
                    "trend_strength_sell": -15,
                    "buy_signal_threshold": 35,
                    "sell_signal_threshold": 35,
                }
            }
        
        self.strategy = Strategy(self.strategy_config)
        
        # Test API connection
        self.test_api_connection()
    
    def test_api_connection(self):
        """Test basic API connectivity."""
        print(f"\n🔍 Testing API Connection...")
        test_url = f"{self.base_url}/v2/account"
        
        try:
            response = requests.get(test_url, headers=self.headers)
            print(f"   Account endpoint: {response.status_code}")
            if response.status_code != 200:
                print(f"   Response: {response.text[:200]}")
        except Exception as e:
            print(f"   Account test failed: {e}")
            
        # Test market data endpoint
        test_url = f"{self.base_url}/v2/stocks/AAPL/bars"
        params = {'start': '2024-01-01', 'end': '2024-01-02', 'timeframe': '1Day'}
        
        try:
            response = requests.get(test_url, headers=self.headers, params=params)
            print(f"   Market data endpoint: {response.status_code}")
            if response.status_code != 200:
                print(f"   Response: {response.text[:200]}")
        except Exception as e:
            print(f"   Market data test failed: {e}")
        print("=")
        
    def get_alpaca_data(self, symbol, start_date, end_date, timeframe='1Day'):
        """Fetch historical data from Alpaca API."""
        try:
            # Use the correct API v2 format
            url = f"{self.base_url}/v2/stocks/{symbol}/bars"
            
            # Convert dates to RFC3339 format (required by Alpaca)
            from datetime import datetime
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            
            params = {
                'start': start_dt.strftime('%Y-%m-%dT00:00:00Z'),
                'end': end_dt.strftime('%Y-%m-%dT23:59:59Z'),
                'timeframe': timeframe,
                'limit': 10000,
                'adjustment': 'raw',
                'asof': '',
                'feed': ''
            }
            
            response = requests.get(url, headers=self.headers, params=params)
            
            if response.status_code != 200:
                print(f"Error {response.status_code}: {response.text[:100]}")
                return None
                
            data = response.json()
            
            if 'bars' not in data or not data['bars']:
                print(f"No data available for {symbol}")
                return None
                
            # Convert to DataFrame
            bars = data['bars']
            df = pd.DataFrame(bars)
            
            # Convert timestamp and set as index
            df['timestamp'] = pd.to_datetime(df['t'])
            df.set_index('timestamp', inplace=True)
            
            # Rename columns to standard format
            df.rename(columns={
                'o': 'open',
                'h': 'high', 
                'l': 'low',
                'c': 'close',
                'v': 'volume'
            }, inplace=True)
            
            # Select only OHLCV columns
            df = df[['open', 'high', 'low', 'close', 'volume']]
            
            return df
            
        except Exception as e:
            print(f"Exception fetching {symbol}: {e}")
            return None
    
    def calculate_performance_metrics(self, returns, benchmark_returns):
        """Calculate comprehensive performance metrics."""
        returns = returns.dropna()
        benchmark_returns = benchmark_returns.dropna()
        
        if len(returns) == 0:
            return None
            
        # Total returns
        total_return = (1 + returns).prod() - 1
        benchmark_total_return = (1 + benchmark_returns).prod() - 1
        
        # Annualized returns
        years = len(returns) / 252
        if years > 0:
            annualized_return = (1 + total_return) ** (1/years) - 1
            benchmark_annualized = (1 + benchmark_total_return) ** (1/years) - 1
        else:
            annualized_return = 0
            benchmark_annualized = 0
            
        # Risk metrics
        volatility = returns.std() * np.sqrt(252)
        
        # Sharpe ratio (2% risk-free rate)
        risk_free_rate = 0.02
        sharpe_ratio = (annualized_return - risk_free_rate) / volatility if volatility > 0 else 0
        
        # Maximum drawdown
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Win rate and other metrics
        win_rate = (returns > 0).mean()
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'win_rate': win_rate,
            'benchmark_return': benchmark_total_return,
            'excess_return': total_return - benchmark_total_return,
            'total_trades': (returns != 0).sum()
        }
    
    def backtest_symbol(self, symbol, start_date, end_date):
        """Backtest strategy on a single symbol."""
        print(f"Testing {symbol}...", end=" ")
        
        # Get data from Alpaca
        data = self.get_alpaca_data(symbol, start_date, end_date)
        
        if data is None or len(data) < 100:
            print("❌ Insufficient data")
            return None
            
        try:
            # Generate signals using our enhanced strategy
            signals = self.strategy.generate_signals(data)
            
            # Calculate returns
            positions = signals['signal'].copy()
            daily_returns = data['close'].pct_change()
            
            # Strategy returns: position * next day's return
            strategy_returns = positions.shift(1) * daily_returns
            buy_hold_returns = daily_returns
            
            # Calculate metrics
            metrics = self.calculate_performance_metrics(strategy_returns, buy_hold_returns)
            
            if metrics:
                metrics['symbol'] = symbol
                metrics['data_points'] = len(data)
                
                # Signal statistics
                metrics['signals'] = {
                    'buy': (signals['signal'] == 1).sum(),
                    'sell': (signals['signal'] == -1).sum(), 
                    'hold': (signals['signal'] == 0).sum()
                }
                
                # Signal strength statistics
                if 'buy_strength' in signals.columns:
                    buy_signals = signals[signals['signal'] == 1]
                    sell_signals = signals[signals['signal'] == -1]
                    
                    metrics['avg_buy_strength'] = buy_signals['buy_strength'].mean() if len(buy_signals) > 0 else 0
                    metrics['avg_sell_strength'] = sell_signals['sell_strength'].mean() if len(sell_signals) > 0 else 0
                    metrics['avg_confidence'] = signals['signal_confidence'].mean()
                
                print(f"✅ Return: {metrics['total_return']:.2%}, Excess: {metrics['excess_return']:.2%}")
                return metrics
            else:
                print("❌ Calculation failed")
                return None
                
        except Exception as e:
            print(f"❌ Error: {e}")
            return None
    
    def run_backtest(self, symbols, start_date, end_date, max_symbols=50):
        """Run comprehensive backtest."""
        print("🚀 ENHANCED S&P 500 TRADING STRATEGY BACKTEST")
        print("=" * 60)
        print(f"📅 Period: {start_date} to {end_date}")
        print(f"📊 Testing: {min(len(symbols), max_symbols)} symbols")
        print(f"⚙️  Strategy: {len(self.strategy_config['indicators'])} indicators, {len(self.strategy_config['patterns'])} patterns")
        print("=" * 60)
        
        results = []
        test_symbols = symbols[:max_symbols]
        
        for i, symbol in enumerate(test_symbols, 1):
            print(f"[{i:2d}/{len(test_symbols)}] ", end="")
            result = self.backtest_symbol(symbol, start_date, end_date)
            
            if result:
                results.append(result)
                
        return pd.DataFrame(results)


def analyze_backtest_results(results_df):
    """Comprehensive analysis of backtest results."""
    if results_df.empty:
        print("❌ No results to analyze!")
        return
        
    print("\n" + "=" * 80)
    print("🎯 COMPREHENSIVE STRATEGY ANALYSIS")
    print("=" * 80)
    
    # Basic statistics
    total_stocks = len(results_df)
    profitable = (results_df['total_return'] > 0).sum()
    outperformed = (results_df['excess_return'] > 0).sum()
    
    print(f"\n📈 PERFORMANCE OVERVIEW:")
    print(f"   Stocks tested: {total_stocks}")
    print(f"   Profitable strategies: {profitable} ({profitable/total_stocks:.1%})")
    print(f"   Outperformed buy & hold: {outperformed} ({outperformed/total_stocks:.1%})")
    
    # Strategy performance
    avg_return = results_df['total_return'].mean()
    avg_benchmark = results_df['benchmark_return'].mean()
    avg_excess = results_df['excess_return'].mean()
    
    print(f"\n💰 RETURNS ANALYSIS:")
    print(f"   Strategy average return: {avg_return:.2%}")
    print(f"   Buy & hold average return: {avg_benchmark:.2%}")
    print(f"   Average excess return: {avg_excess:.2%}")
    print(f"   Best performer: {results_df.loc[results_df['total_return'].idxmax(), 'symbol']} ({results_df['total_return'].max():.2%})")
    print(f"   Worst performer: {results_df.loc[results_df['total_return'].idxmin(), 'symbol']} ({results_df['total_return'].min():.2%})")
    
    # Risk metrics
    print(f"\n⚠️  RISK ANALYSIS:")
    print(f"   Average Sharpe ratio: {results_df['sharpe_ratio'].mean():.3f}")
    print(f"   Average volatility: {results_df['volatility'].mean():.2%}")
    print(f"   Average max drawdown: {results_df['max_drawdown'].mean():.2%}")
    print(f"   Average win rate: {results_df['win_rate'].mean():.1%}")
    
    # Signal analysis
    total_buy_signals = sum([result['signals']['buy'] for result in results_df.to_dict('records')])
    total_sell_signals = sum([result['signals']['sell'] for result in results_df.to_dict('records')])
    avg_trades = results_df['total_trades'].mean()
    
    print(f"\n🎯 SIGNAL ANALYSIS:")
    print(f"   Total buy signals: {total_buy_signals:,}")
    print(f"   Total sell signals: {total_sell_signals:,}")
    print(f"   Average trades per stock: {avg_trades:.1f}")
    
    if 'avg_buy_strength' in results_df.columns:
        print(f"   Average buy signal strength: {results_df['avg_buy_strength'].mean():.1f}")
        print(f"   Average sell signal strength: {results_df['avg_sell_strength'].mean():.1f}")
        print(f"   Average signal confidence: {results_df['avg_confidence'].mean():.1f}")
    
    # Top performers
    print(f"\n🏆 TOP 10 PERFORMERS:")
    top_performers = results_df.nlargest(10, 'total_return')
    for _, row in top_performers.iterrows():
        print(f"   {row['symbol']:5} | {row['total_return']:7.2%} | Excess: {row['excess_return']:7.2%} | Sharpe: {row['sharpe_ratio']:6.3f}")
    
    # Strategy effectiveness
    print(f"\n📊 STRATEGY EFFECTIVENESS:")
    strong_outperformers = (results_df['excess_return'] > 0.05).sum()  # >5% excess return
    strong_underperformers = (results_df['excess_return'] < -0.05).sum()  # <-5% excess return
    
    print(f"   Strong outperformers (>5% excess): {strong_outperformers} ({strong_outperformers/total_stocks:.1%})")
    print(f"   Strong underperformers (<-5% excess): {strong_underperformers} ({strong_underperformers/total_stocks:.1%})")
    
    # Overall assessment
    print(f"\n🎖️  OVERALL STRATEGY ASSESSMENT:")
    if avg_excess > 0:
        print(f"   ✅ Strategy shows positive average excess return of {avg_excess:.2%}")
    else:
        print(f"   ⚠️  Strategy shows negative average excess return of {avg_excess:.2%}")
        
    if outperformed / total_stocks > 0.5:
        print(f"   ✅ Strategy outperforms buy & hold on majority of stocks ({outperformed/total_stocks:.1%})")
    else:
        print(f"   ⚠️  Strategy underperforms buy & hold on majority of stocks ({outperformed/total_stocks:.1%})")
    
    return results_df


def main():
    """Main execution function."""
    # Configuration file path
    config_path = "../trading_bot_project/config.yaml"
    
    # Initialize backtester with your Alpaca credentials
    backtester = AlpacaBacktester(config_path)
    
    # Set date range (use older data that should be available with basic subscription)
    end_date = datetime.now() - timedelta(days=30)  # End 30 days ago
    start_date = end_date - timedelta(days=365)     # 1 year of data, ending 30 days ago
    
    print(f"📅 Using historical data to avoid subscription restrictions")
    print(f"📅 Data period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    # Run backtest on top 20 S&P 500 companies (reasonable speed)
    results_df = backtester.run_backtest(
        SP500_TOP_100,
        start_date.strftime('%Y-%m-%d'),
        end_date.strftime('%Y-%m-%d'),
        max_symbols=20
    )
    
    # Analyze results
    if not results_df.empty:
        final_results = analyze_backtest_results(results_df)
        
        # Save results
        results_df.to_csv('alpaca_sp500_backtest_results.csv', index=False)
        print(f"\n💾 Results saved to 'alpaca_sp500_backtest_results.csv'")
        
        return final_results
    else:
        print("❌ No successful backtests completed!")
        return None


if __name__ == "__main__":
    results = main()
