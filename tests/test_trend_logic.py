"""Test script to verify the improved higher_lows and lower_highs logic."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trading_bot_project.indicators import higher_lows, lower_highs

def test_higher_lows_lower_highs():
    """Test the improved logic with a synthetic trend pattern."""
    
    # Create synthetic data with clear higher lows and lower highs patterns
    dates = pd.date_range('2024-01-01', periods=50, freq='D')
    
    # Create an uptrend with higher lows (first 25 days)
    uptrend_lows = [10, 8, 12, 9, 14, 11, 16, 13, 18, 15, 20, 17, 22, 19, 24, 21, 26, 23, 28, 25, 30, 27, 32, 29, 34]
    
    # Create a downtrend with lower highs (last 25 days)  
    downtrend_highs = [35, 38, 33, 36, 31, 34, 29, 32, 27, 30, 25, 28, 23, 26, 21, 24, 19, 22, 17, 20, 15, 18, 13, 16, 11]
    
    # Combine into full series
    lows = uptrend_lows + downtrend_highs
    highs = [x + 5 for x in lows]  # Highs are always 5 points above lows
    
    # Create DataFrame
    df = pd.DataFrame({
        'low': lows,
        'high': highs,
        'close': [(low + high) / 2 for low, high in zip(lows, highs)]
    }, index=dates)
    
    # Test higher lows
    higher_lows_result = higher_lows(df['low'], window=2)
    lower_highs_result = lower_highs(df['high'], window=2)
    
    print("=== Testing Improved Higher Lows and Lower Highs Logic ===\n")
    
    print("Sample data (first 10 rows):")
    print(df.head(10))
    print()
    
    print("Higher Lows detected:")
    higher_lows_dates = df.index[higher_lows_result]
    for date in higher_lows_dates:
        idx = df.index.get_loc(date)
        print(f"  {date.strftime('%Y-%m-%d')}: Low = {df.loc[date, 'low']:.1f}")
    print(f"Total higher lows: {higher_lows_result.sum()}")
    print()
    
    print("Lower Highs detected:")
    lower_highs_dates = df.index[lower_highs_result]
    for date in lower_highs_dates:
        idx = df.index.get_loc(date)
        print(f"  {date.strftime('%Y-%m-%d')}: High = {df.loc[date, 'high']:.1f}")
    print(f"Total lower highs: {lower_highs_result.sum()}")
    print()
    
    # Verify logic manually for a few key points
    print("=== Manual Verification ===")
    
    # Check first part should have higher lows
    first_half_higher_lows = higher_lows_result[:25].sum()
    second_half_higher_lows = higher_lows_result[25:].sum()
    print(f"Higher lows in first half (uptrend): {first_half_higher_lows}")
    print(f"Higher lows in second half (downtrend): {second_half_higher_lows}")
    
    # Check second part should have lower highs
    first_half_lower_highs = lower_highs_result[:25].sum()
    second_half_lower_highs = lower_highs_result[25:].sum()
    print(f"Lower highs in first half (uptrend): {first_half_lower_highs}")
    print(f"Lower highs in second half (downtrend): {second_half_lower_highs}")
    
    print("\n=== Results Analysis ===")
    if first_half_higher_lows > 0:
        print("✅ GOOD: Higher lows detected in uptrend section")
    else:
        print("❌ ISSUE: No higher lows detected in uptrend section")
        
    if second_half_lower_highs > 0:
        print("✅ GOOD: Lower highs detected in downtrend section")
    else:
        print("❌ ISSUE: No lower highs detected in downtrend section")
        
    # Test with simple manual example
    print("\n=== Simple Manual Test ===")
    simple_lows = pd.Series([10, 8, 12, 9, 14, 11, 16], index=range(7))
    simple_higher_lows = higher_lows(simple_lows, window=1)
    
    print("Simple lows:", simple_lows.tolist())
    print("Higher lows detected at indices:", simple_higher_lows[simple_higher_lows].index.tolist())
    
    # Should detect higher lows at indices where the low is higher than the previous local low
    # Local lows should be at indices 1 (8), 3 (9), 5 (11)
    # Higher lows should be at indices 3 (9 > 8) and 5 (11 > 9)
    expected_indices = [3, 5]  # These should be higher lows
    actual_indices = simple_higher_lows[simple_higher_lows].index.tolist()
    
    if set(actual_indices) == set(expected_indices):
        print("✅ PERFECT: Simple test matches expected results")
    else:
        print(f"⚠️  PARTIAL: Expected {expected_indices}, got {actual_indices}")
    
    return True

if __name__ == "__main__":
    test_higher_lows_lower_highs()
