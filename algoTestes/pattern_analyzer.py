import pandas as pd
import numpy as np
from collections import defaultdict

def analyze_patterns(df):
    """
    Analyze various patterns in the financial data and return detailed statistics.
    
    Parameters:
    df (pandas.DataFrame): DataFrame containing financial data with columns:
                          'Date', 'Change %', etc.
    
    Returns:
    dict: Dictionary containing various pattern analyses
    """
    # Ensure data is properly formatted
    if 'Date' not in df.columns or 'Change %' not in df.columns:
        return {"error": "Required columns not found in data"}
    
    # Make a copy to avoid modifying the original dataframe
    df = df.copy()
    
    # Ensure date is datetime and sort chronologically
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    
    # Initialize results dictionary
    results = {}
    
    # Calculate weekly changes
    df['Week_Number'] = df['Date'].dt.isocalendar().week
    df['Year'] = df['Date'].dt.isocalendar().year
    weekly_changes = df.groupby(['Year', 'Week_Number'])['Change %'].sum().reset_index()
    weekly_changes = weekly_changes.sort_values(['Year', 'Week_Number'])
    
    # 1. Consecutive Weeks Analysis
    consecutive_weeks_analysis = analyze_consecutive_weeks(weekly_changes)
    results['consecutive_weeks'] = consecutive_weeks_analysis
    
    # Day of Week Analysis has been removed
    
    # 3. Monthly Analysis
    df['Month'] = df['Date'].dt.month
    monthly_analysis = analyze_monthly_patterns(df)
    results['monthly'] = monthly_analysis
    
    # 4. Volatility Clustering Analysis
    volatility_analysis = analyze_volatility_clustering(df)
    results['volatility'] = volatility_analysis
    
    # Price Level Analysis has been removed
    
    return results

def analyze_consecutive_weeks(weekly_changes):
    """
    Analyze patterns following consecutive positive or negative weeks.
    
    Parameters:
    weekly_changes (pandas.DataFrame): DataFrame with weekly change data
    
    Returns:
    dict: Statistics about what happens after consecutive weeks
    """
    results = {}
    
    # Get the list of weekly changes
    changes = weekly_changes['Change %'].values
    
    # Initialize data structures for tracking streaks and outcomes
    streaks = []
    current_streak = {'type': None, 'length': 0, 'start_idx': 0}
    
    # First pass: identify all streaks
    for i in range(len(changes)):
        current_change = changes[i]
        
        # Determine streak type and length
        if current_change > 0:
            if current_streak['type'] == 'positive':
                current_streak['length'] += 1
            else:
                # End previous streak if it exists
                if current_streak['type'] is not None and current_streak['length'] > 0:
                    streaks.append(current_streak)
                # Start new positive streak
                current_streak = {'type': 'positive', 'length': 1, 'start_idx': i}
        elif current_change < 0:
            if current_streak['type'] == 'negative':
                current_streak['length'] += 1
            else:
                # End previous streak if it exists
                if current_streak['type'] is not None and current_streak['length'] > 0:
                    streaks.append(current_streak)
                # Start new negative streak
                current_streak = {'type': 'negative', 'length': 1, 'start_idx': i}
        else:  # current_change == 0
            # End previous streak if it exists
            if current_streak['type'] is not None and current_streak['length'] > 0:
                streaks.append(current_streak)
            # Reset streak (neutral weeks break streaks)
            current_streak = {'type': None, 'length': 0, 'start_idx': i+1}
    
    # Add the last streak if it exists
    if current_streak['type'] is not None and current_streak['length'] > 0:
        streaks.append(current_streak)
    
    # Initialize counters for different streak lengths
    streak_stats = {
        'positive': defaultdict(lambda: {'count': 0, 'next_positive': 0, 'next_negative': 0, 'next_avg_change': []}),
        'negative': defaultdict(lambda: {'count': 0, 'next_positive': 0, 'next_negative': 0, 'next_avg_change': []})
    }
    
    # Second pass: analyze what happens after each streak
    for streak in streaks:
        streak_type = streak['type']
        streak_length = streak['length']
        end_idx = streak['start_idx'] + streak_length - 1
        
        # Check if there's a next week after this streak
        if end_idx + 1 < len(changes):
            next_change = changes[end_idx + 1]
            
            # Only count the maximum streak length for each streak
            # This ensures we don't double-count shorter streaks within longer ones
            length = min(streak_length, 10)  # Cap at 10 for analysis
            streak_stats[streak_type][length]['count'] += 1
            
            if next_change > 0:
                streak_stats[streak_type][length]['next_positive'] += 1
            elif next_change < 0:
                streak_stats[streak_type][length]['next_negative'] += 1
            
            streak_stats[streak_type][length]['next_avg_change'].append(next_change)
    
    # Calculate probabilities and average changes
    for streak_type in ['positive', 'negative']:
        for streak_length in range(1, 11):  # Analyze streaks of length 1 to 10
            stats = streak_stats[streak_type][streak_length]
            
            if stats['count'] > 0:
                # Calculate probabilities
                prob_positive = (stats['next_positive'] / stats['count']) * 100
                prob_negative = (stats['next_negative'] / stats['count']) * 100
                
                # Calculate average next change
                avg_next_change = sum(stats['next_avg_change']) / len(stats['next_avg_change']) if stats['next_avg_change'] else 0
                
                results[f"{streak_length}_{streak_type}"] = {
                    'count': stats['count'],
                    'prob_next_positive': round(prob_positive, 2),
                    'prob_next_negative': round(prob_negative, 2),
                    'avg_next_change': round(avg_next_change, 2)
                }
    
    return results



def analyze_monthly_patterns(df):
    """
    Analyze patterns for each month of the year.
    
    Parameters:
    df (pandas.DataFrame): DataFrame with daily data
    
    Returns:
    dict: Statistics about each month
    """
    results = {}
    
    # Ensure we have the month
    if 'Month' not in df.columns:
        df['Month'] = df['Date'].dt.month
    
    # Add month name for readability
    month_names = {
        1: 'January', 2: 'February', 3: 'March', 4: 'April',
        5: 'May', 6: 'June', 7: 'July', 8: 'August',
        9: 'September', 10: 'October', 11: 'November', 12: 'December'
    }
    df['Month_Name'] = df['Month'].map(month_names)
    
    # Calculate monthly statistics
    monthly_stats = df.groupby('Month_Name')['Change %'].agg([
        ('mean', 'mean'),
        ('median', 'median'),
        ('positive', lambda x: (x > 0).mean() * 100),
        ('negative', lambda x: (x < 0).mean() * 100),
        ('count', 'count')
    ])
    
    # Calculate monthly returns (sum of daily changes within each month)
    df['Year_Month'] = df['Date'].dt.strftime('%Y-%m')
    monthly_returns = df.groupby(['Year_Month', 'Month_Name'])['Change %'].sum().reset_index()
    
    # Calculate statistics for each month across years
    month_performance = {}
    for month in month_names.values():
        month_data = monthly_returns[monthly_returns['Month_Name'] == month]['Change %']
        if not month_data.empty:
            month_performance[month] = {
                'avg_monthly_return': round(month_data.mean(), 2),
                'positive_months': round((month_data > 0).mean() * 100, 2),
                'negative_months': round((month_data < 0).mean() * 100, 2),
                'count': len(month_data)
            }
    
    # Format monthly stats for return
    monthly_results = {}
    for month in month_names.values():
        if month in monthly_stats.index:
            monthly_results[month] = {
                'mean_daily_change': round(monthly_stats.loc[month, 'mean'], 2),
                'median_daily_change': round(monthly_stats.loc[month, 'median'], 2),
                'positive_days_pct': round(monthly_stats.loc[month, 'positive'], 2),
                'negative_days_pct': round(monthly_stats.loc[month, 'negative'], 2),
                'day_count': int(monthly_stats.loc[month, 'count'])
            }
    
    return {
        'daily_stats': monthly_results,
        'monthly_performance': month_performance
    }

def analyze_volatility_clustering(df):
    """
    Analyze volatility clustering patterns.
    
    Parameters:
    df (pandas.DataFrame): DataFrame with daily data
    
    Returns:
    dict: Statistics about volatility patterns
    """
    results = {}
    
    # Calculate absolute change as a measure of volatility
    df['Abs_Change'] = df['Change %'].abs()
    
    # Define high volatility as days with absolute change above the 75th percentile
    volatility_threshold = df['Abs_Change'].quantile(0.75)
    df['High_Volatility'] = df['Abs_Change'] > volatility_threshold
    
    # Calculate probability of high volatility following high volatility
    high_vol_days = df[df['High_Volatility']].index
    next_day_high_vol_count = 0
    
    for idx in high_vol_days:
        if idx + 1 in df.index and df.loc[idx + 1, 'High_Volatility']:
            next_day_high_vol_count += 1
    
    if len(high_vol_days) > 0:
        prob_high_vol_after_high_vol = (next_day_high_vol_count / len(high_vol_days)) * 100
        results['volatility_clustering'] = {
            'threshold': round(volatility_threshold, 2),
            'prob_high_vol_after_high_vol': round(prob_high_vol_after_high_vol, 2),
            'high_vol_days_count': len(high_vol_days),
            'consecutive_high_vol_count': next_day_high_vol_count
        }
    
    # Analyze what happens after volatile days
    after_high_vol = {
        'positive': {'count': 0, 'avg_change': []},
        'negative': {'count': 0, 'avg_change': []}
    }
    
    for idx in high_vol_days:
        if idx + 1 in df.index:
            next_change = df.loc[idx + 1, 'Change %']
            
            if next_change > 0:
                after_high_vol['positive']['count'] += 1
                after_high_vol['positive']['avg_change'].append(next_change)
            elif next_change < 0:
                after_high_vol['negative']['count'] += 1
                after_high_vol['negative']['avg_change'].append(next_change)
    
    total_after_high_vol = after_high_vol['positive']['count'] + after_high_vol['negative']['count']
    
    if total_after_high_vol > 0:
        prob_positive_after_high_vol = (after_high_vol['positive']['count'] / total_after_high_vol) * 100
        
        avg_positive_after_high_vol = (
            sum(after_high_vol['positive']['avg_change']) / len(after_high_vol['positive']['avg_change'])
            if after_high_vol['positive']['avg_change'] else 0
        )
        
        avg_negative_after_high_vol = (
            sum(after_high_vol['negative']['avg_change']) / len(after_high_vol['negative']['avg_change'])
            if after_high_vol['negative']['avg_change'] else 0
        )
        
        results['after_high_volatility'] = {
            'prob_positive': round(prob_positive_after_high_vol, 2),
            'prob_negative': round(100 - prob_positive_after_high_vol, 2),
            'avg_positive_change': round(avg_positive_after_high_vol, 2),
            'avg_negative_change': round(avg_negative_after_high_vol, 2)
        }
    
    return results



def format_pattern_results(results):
    """
    Format the pattern analysis results into a readable string.
    
    Parameters:
    results (dict): The results from analyze_patterns
    
    Returns:
    str: Formatted string with analysis results
    """

    output = """
<h>Pattern Analysis Report</h>
"""
    # 1. Consecutive Weeks Analysis
    output += "Consecutive Weeks Patterns:\n"
    output += "===========================\n"
    
    if 'consecutive_weeks' in results:
        # First, collect all streak patterns that have at least one occurrence
        valid_patterns = []
        for streak_length in range(1, 11):  # Analyze streaks of length 1 to 10
            pos_key = f"{streak_length}_positive"
            neg_key = f"{streak_length}_negative"
            
            if pos_key in results['consecutive_weeks']:
                valid_patterns.append((streak_length, 'positive'))
            if neg_key in results['consecutive_weeks']:
                valid_patterns.append((streak_length, 'negative'))
        
        if valid_patterns:
            # Create a table header
            output += "| STREAK PATTERN                | OCCURRENCES | NEXT WEEK AVG CHANGE |\n"
            output += "|-------------------------------|------------|---------------------|\n"
            
            # Display each valid pattern
            for streak_length, streak_type in valid_patterns:
                key = f"{streak_length}_{streak_type}"
                stats = results['consecutive_weeks'][key]
                
                pattern_name = f"{streak_length} consecutive {streak_type} {'week' if streak_length == 1 else 'weeks'}"
                occurrences = stats['count']
                avg_change = stats['avg_next_change']
                
                output += f"| {pattern_name:<29} | {occurrences:^10} | {avg_change:^19.2f}% |\n"
            
            output += "|-------------------------------|------------|---------------------|\n"
        else:
            output += "No consecutive week patterns found in the data.\n"
    
    # 3. Monthly Patterns
    output += "Monthly Patterns:\n"
    output += "================\n"
    
    if 'monthly' in results and 'monthly_performance' in results['monthly']:
        month_order = [
            'January', 'February', 'March', 'April', 'May', 'June',
            'July', 'August', 'September', 'October', 'November', 'December'
        ]
        
        for month in month_order:
            if month in results['monthly']['monthly_performance']:
                stats = results['monthly']['monthly_performance'][month]
                output += f"{month}:\n"
                output += f"  Average monthly return: {stats['avg_monthly_return']}%\n"
                output += f"  Positive months: {stats['positive_months']}%\n"
                output += f"  Negative months: {stats['negative_months']}%\n"
                output += f"  Sample size: {stats['count']} months\n\n"
    
    # 4. Volatility Clustering
    output += "Volatility Patterns:\n"
    output += "===================\n"
    
    if 'volatility' in results:
        if 'volatility_clustering' in results['volatility']:
            stats = results['volatility']['volatility_clustering']
            output += f"Volatility threshold (75th percentile): {stats['threshold']}%\n"
            output += f"Probability of high volatility after high volatility: {stats['prob_high_vol_after_high_vol']}%\n"
            output += f"Number of high volatility days: {stats['high_vol_days_count']}\n"
            output += f"Number of consecutive high volatility days: {stats['consecutive_high_vol_count']}\n\n"
        
        if 'after_high_volatility' in results['volatility']:
            stats = results['volatility']['after_high_volatility']
            output += "After high volatility days:\n"
            output += f"  Probability of positive day: {stats['prob_positive']}%\n"
            output += f"  Probability of negative day: {stats['prob_negative']}%\n"
            output += f"  Average positive change: {stats['avg_positive_change']}%\n"
            output += f"  Average negative change: {stats['avg_negative_change']}%\n\n"
    
    # Price Level Analysis section has been removed
    
    return output
