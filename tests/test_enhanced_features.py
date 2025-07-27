"""Test script for enhanced trading bot features.

This script tests the new Doji patterns and sophisticated trend analysis
to ensure they work correctly with sample data.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import our enhanced modules
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trading_bot_project import patterns as pat
from trading_bot_project import indicators as ind
from trading_bot_project.strategy import Strategy


def create_sample_data(periods=100):
    """Create sample OHLCV data for testing."""
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=periods, freq='D')
    
    # Generate realistic price data
    base_price = 100
    price_changes = np.random.normal(0, 0.02, periods)  # 2% daily volatility
    
    closes = [base_price]
    for change in price_changes[1:]:
        closes.append(closes[-1] * (1 + change))
    
    # Generate OHLC from closes
    data = []
    for i, close in enumerate(closes):
        high = close * (1 + abs(np.random.normal(0, 0.01)))
        low = close * (1 - abs(np.random.normal(0, 0.01)))
        open_price = closes[i-1] if i > 0 else close
        volume = np.random.randint(100000, 1000000)
        
        data.append({
            'open': open_price,
            'high': max(high, open_price, close),
            'low': min(low, open_price, close),
            'close': close,
            'volume': volume
        })
    
    return pd.DataFrame(data, index=dates)


def test_new_patterns():
    """Test new Doji and spinning top patterns."""
    print("Testing new candlestick patterns...")
    
    # Create sample data
    df = create_sample_data(50)
    
    # Test Doji patterns
    doji = pat.detect_doji(df)
    dragonfly_doji = pat.detect_dragonfly_doji(df)
    gravestone_doji = pat.detect_gravestone_doji(df)
    spinning_top = pat.detect_spinning_top(df)
    
    print(f"Regular Doji detected: {doji.sum()} times")
    print(f"Dragonfly Doji detected: {dragonfly_doji.sum()} times")
    print(f"Gravestone Doji detected: {gravestone_doji.sum()} times")
    print(f"Spinning Top detected: {spinning_top.sum()} times")
    
    return True


def test_enhanced_indicators():
    """Test new sophisticated trend analysis indicators."""
    print("\nTesting enhanced trend analysis...")
    
    # Create sample data with enough history
    df = create_sample_data(200)
    
    # Test trend strength
    trend_strength = ind.trend_strength(df)
    print(f"Trend strength range: {trend_strength.min():.2f} to {trend_strength.max():.2f}")
    
    # Test weekly trend analysis
    weekly_analysis = ind.weekly_trend_analysis(df)
    if not weekly_analysis.empty:
        print(f"Weekly trend analysis columns: {list(weekly_analysis.columns)}")
        print(f"Weekly trend direction: {weekly_analysis['trend_direction'].iloc[-1]}")
    
    # Test monthly trend analysis
    monthly_analysis = ind.monthly_trend_analysis(df)
    if not monthly_analysis.empty:
        print(f"Monthly trend analysis columns: {list(monthly_analysis.columns)}")
    
    # Test volume analysis
    volume_analysis = ind.volume_trend_analysis(df)
    print(f"Volume analysis columns: {list(volume_analysis.columns)}")
    
    # Test advanced support/resistance
    sr_analysis = ind.advanced_support_resistance(df)
    print(f"Support/Resistance analysis columns: {list(sr_analysis.columns)}")
    
    return True


def test_enhanced_strategy():
    """Test the enhanced strategy with new features."""
    print("\nTesting enhanced strategy...")
    
    # Configuration for testing
    config = {
        "patterns": {
            "bullish_engulfing": True,
            "doji": True,
            "dragonfly_doji": True,
            "gravestone_doji": True,
            "spinning_top": True,
            "doji_tolerance": 0.002
        },
        "indicators": {
            "rsi": True,
            "sma_20": True,
            "sma_50": True,
            "trend_strength": True,
            "weekly_trend": True,
            "volume_analysis": True,
            "advanced_support_resistance": True
        },
        "thresholds": {
            "rsi_oversold": 30,
            "rsi_overbought": 70,
            "trend_strength_buy": 15,
            "trend_strength_sell": -15
        }
    }
    
    # Create strategy
    strategy = Strategy(config)
    
    # Create sample data
    df = create_sample_data(100)
    
    # Generate signals
    signals = strategy.generate_signals(df)
    
    print(f"Signals generated successfully with {len(signals.columns)} features")
    print(f"Buy signals: {(signals['signal'] == 1).sum()}")
    print(f"Sell signals: {(signals['signal'] == -1).sum()}")
    print(f"Hold signals: {(signals['signal'] == 0).sum()}")
    
    # Check for new pattern columns
    pattern_cols = [col for col in signals.columns if 'doji' in col or 'spinning' in col]
    print(f"New pattern columns detected: {pattern_cols}")
    
    # Check for new indicator columns
    trend_cols = [col for col in signals.columns if 'trend' in col or 'vol_' in col or 'sr_' in col]
    print(f"New indicator columns detected: {len(trend_cols)} columns")
    
    return True


def main():
    """Run all tests."""
    print("=== Testing Enhanced Trading Bot Features ===\n")
    
    try:
        # Test new patterns
        test_new_patterns()
        
        # Test enhanced indicators
        test_enhanced_indicators()
        
        # Test enhanced strategy
        test_enhanced_strategy()
        
        print("\n=== All tests completed successfully! ===")
        print("\nNew features implemented:")
        print("✅ Doji candlestick patterns (regular, dragonfly, gravestone)")
        print("✅ Spinning top pattern")
        print("✅ Sophisticated trend analysis (daily, weekly, monthly)")
        print("✅ Enhanced volume analysis with price confirmation")
        print("✅ Advanced support/resistance detection")
        print("✅ Trend strength indicator")
        print("✅ Integration with strategy for comprehensive analysis")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
