import pandas as pd
import numpy as np
from collections import defaultdict

def analyze_avg_range(df):
    """
    Analyze the average range blocks for the 'Change %' values in the dataframe.
    
    Parameters:
    df (pandas.DataFrame): DataFrame containing financial data with a 'Change %' column
    
    Returns:
    dict: Dictionary containing the analysis results
    """
    # Initialize results dictionary
    results = {}
    
    # Ensure the required column exists
    if 'Change %' not in df.columns:
        return {"error": "Required column 'Change %' not found in data"}
    
    # Make a copy to avoid modifying the original dataframe
    df = df.copy()
    
    # Ensure date is datetime and sort chronologically
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
    
    # Create a column for the range blocks (0.30% blocks)
    df['Range_Block'] = df['Change %'].apply(lambda x: calculate_range_block(x))
    
    # Group by range blocks
    range_blocks = df.groupby('Range_Block')
    
    # Initialize data structure for range block analysis
    range_block_analysis = {}
    
    # Analyze each range block
    for block, block_data in range_blocks:
        # Skip if block is None or NaN
        if block is None or pd.isna(block):
            continue
        
        # Calculate average of the 'Change %' values in this block
        avg_change = block_data['Change %'].mean()
        
        # Get dates for this block
        dates = block_data['Date'].tolist() if 'Date' in block_data.columns else []
        
        # Get week information if available
        weeks = []
        if 'Date' in block_data.columns:
            block_data['Week'] = block_data['Date'].dt.isocalendar().week
            block_data['Year'] = block_data['Date'].dt.isocalendar().year
            block_data['Week_ID'] = block_data['Year'].astype(str) + '-' + block_data['Week'].astype(str)
            weeks = block_data['Week_ID'].unique().tolist()
        
        # Identify streaks
        streaks = identify_streaks(block_data) if 'Date' in block_data.columns else []
        
        # Store analysis for this block
        range_block_analysis[block] = {
            'count': len(block_data),
            'avg_change': round(avg_change, 2),
            'weeks': weeks,
            'streaks': streaks
        }
    
    # Store the range block analysis in the results
    results['range_blocks'] = range_block_analysis
    
    return results

def calculate_range_block(change_percent):
    """
    Calculate the range block for a given change percentage.
    Range blocks are in increments of 0.30%.
    
    Parameters:
    change_percent (float): The change percentage value
    
    Returns:
    float: The range block value (e.g., 0.30, 0.60, 0.90, etc.)
    """
    if pd.isna(change_percent):
        return None
    
    # Handle both positive and negative values
    sign = 1 if change_percent >= 0 else -1
    abs_change = abs(change_percent)
    
    # Calculate the block number (how many 0.30% blocks)
    block_number = int(abs_change / 0.30)
    
    # If the change is exactly at a block boundary, use that block
    if abs_change % 0.30 == 0:
        block_value = block_number * 0.30
    else:
        # Otherwise, go to the next block
        block_value = (block_number + 1) * 0.30
    
    # Apply the sign to get the correct block
    return round(sign * block_value, 2)

def identify_streaks(block_data):
    """
    Identify consecutive day streaks in the data.
    
    Parameters:
    block_data (pandas.DataFrame): DataFrame containing data for a specific range block
    
    Returns:
    list: List of streaks with information about each streak
    """
    if 'Date' not in block_data.columns or len(block_data) <= 1:
        return []
    
    # Sort by date
    block_data = block_data.sort_values('Date')
    
    # Convert dates to days since epoch for easy consecutive day detection
    block_data['Day_Number'] = (block_data['Date'] - pd.Timestamp('1970-01-01')).dt.days
    
    # Initialize variables for streak detection
    streaks = []
    current_streak = []
    prev_day = None
    
    for _, row in block_data.iterrows():
        current_day = row['Day_Number']
        
        if prev_day is None or current_day == prev_day + 1:
            # Continue the streak
            current_streak.append(row)
        else:
            # End the previous streak if it exists and has more than 1 day
            if len(current_streak) > 1:
                streak_info = {
                    'length': len(current_streak),
                    'start_date': current_streak[0]['Date'].strftime('%Y-%m-%d'),
                    'end_date': current_streak[-1]['Date'].strftime('%Y-%m-%d'),
                    'week_ids': list(set([f"{d['Date'].year}-{d['Date'].week}" for d in current_streak]))
                }
                streaks.append(streak_info)
            
            # Start a new streak
            current_streak = [row]
        
        prev_day = current_day
    
    # Don't forget to add the last streak if it exists and has more than 1 day
    if len(current_streak) > 1:
        streak_info = {
            'length': len(current_streak),
            'start_date': current_streak[0]['Date'].strftime('%Y-%m-%d'),
            'end_date': current_streak[-1]['Date'].strftime('%Y-%m-%d'),
            'week_ids': list(set([f"{d['Date'].year}-{d['Date'].week}" for d in current_streak]))
        }
        streaks.append(streak_info)
    
    return streaks

def format_avg_range_results(results):
    """
    Format the average range analysis results into HTML format.
    
    Parameters:
    results (dict): The results from analyze_avg_range
    
    Returns:
    str: Formatted HTML string with analysis results
    """
    if 'error' in results:
        return f"<p class='error'>{results['error']}</p>"
    
    # Start with a back button and heading
    output = """
    <div class="back-button">
        <a href="javascript:history.back()" class="btn">Back to Previous Analysis</a>
    </div>
    <h2>Average Range Block Analysis</h2>
    <p>Analysis of 'Change %' values grouped into 0.30% blocks</p>
    """
    
    # Create a table for the results
    output += "<table class='avg-range-table'>\n"
    
    # Table headers
    output += "<thead>\n<tr>\n"
    headers = ["Range Block", "Avg. Range", "Count", "Streaks", "Weeks"]
    for header in headers:
        output += f"<th>{header}</th>\n"
    output += "</tr>\n</thead>\n"
    
    # Table body
    output += "<tbody>\n"
    
    # Sort range blocks numerically
    if 'range_blocks' in results:
        sorted_blocks = sorted(results['range_blocks'].keys(), 
                              key=lambda x: (abs(x), -1 if x < 0 else 1))
        
        for block in sorted_blocks:
            block_data = results['range_blocks'][block]
            output += "<tr>\n"
            
            # Range Block column
            sign = '+' if block >= 0 else ''
            output += f"<td class='{get_color_class(block)}'>{sign}{block}%</td>\n"
            
            # Avg. Range column
            avg_change = block_data['avg_change']
            sign_avg = '+' if avg_change >= 0 else ''
            output += f"<td class='{get_color_class(avg_change)}'>{sign_avg}{avg_change}%</td>\n"
            
            # Count column
            output += f"<td>{block_data['count']}</td>\n"
            
            # Streaks column
            if block_data['streaks']:
                # Format dates to show only last two digits of year
                streak_text = ", ".join([
                    f"{s['length']} days ({format_short_date(s['start_date'])} to {format_short_date(s['end_date'])})" 
                    for s in block_data['streaks']
                ])
                output += f"<td>{streak_text}</td>\n"
            else:
                output += "<td>No streaks</td>\n"
            
            # Weeks column
            if block_data['weeks']:
                # Group weeks by year
                weeks_by_year = {}
                for week_id in block_data['weeks']:
                    year, week_num = week_id.split('-')
                    if year not in weeks_by_year:
                        weeks_by_year[year] = []
                    weeks_by_year[year].append(week_num)
                
                # Format the weeks text with each year on a new line
                weeks_parts = []
                for year, week_nums in sorted(weeks_by_year.items()):
                    # Sort week numbers numerically
                    sorted_weeks = sorted(map(int, week_nums))
                    weeks_parts.append(f"{year}: {', '.join(map(str, sorted_weeks))}")
                
                # Join with line breaks for HTML display
                weeks_text = "<br>".join(weeks_parts)
                output += f"<td>{weeks_text}</td>\n"
            else:
                output += "<td>-</td>\n"
            
            output += "</tr>\n"
    
    output += "</tbody>\n</table>\n"
    
    # Add CSS for the table
    output += """
    <style>
        .back-button {
            margin-bottom: 20px;
        }
        .btn {
            display: inline-block;
            padding: 8px 16px;
            background-color: #f8f9fa;
            border: 1px solid #ddd;
            border-radius: 4px;
            text-decoration: none;
            color: #333;
        }
        .btn:hover {
            background-color: #e9ecef;
        }
        .avg-range-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
            font-family: Arial, sans-serif;
        }
        .avg-range-table th, .avg-range-table td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }
        .avg-range-table th {
            background-color: #f2f2f2;
            font-weight: bold;
            text-align: center;
        }
        .avg-range-table tr:nth-child(even) {
            background-color: #f9f9f9;
        }
        .avg-range-table tr:hover {
            background-color: #f5f5f5;
        }
        .positive {
            color: green;
        }
        .negative {
            color: red;
        }
        .neutral {
            color: gray;
        }
        .error {
            color: red;
            font-weight: bold;
        }
    </style>
    """
    
    return output

def format_short_date(date_str):
    """
    Format a date string to show only the last two digits of the year.
    
    Parameters:
    date_str (str): Date string in format 'YYYY-MM-DD'
    
    Returns:
    str: Date string with shortened year 'YY-MM-DD'
    """
    if not date_str or len(date_str) < 10:
        return date_str
    
    # Extract year, month, and day
    year = date_str[:4]
    rest = date_str[4:]
    
    # Use only the last two digits of the year
    short_year = year[-2:]
    
    return short_year + rest

def get_color_class(value):
    """
    Get the CSS color class based on the value.
    
    Parameters:
    value (float): The value to determine the color class for
    
    Returns:
    str: CSS class name ('positive', 'negative', or 'neutral')
    """
    if value > 0:
        return 'positive'
    elif value < 0:
        return 'negative'
    else:
        return 'neutral'
