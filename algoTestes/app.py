
from flask import Flask, request, redirect, render_template_string
import pandas as pd
import os
import pattern_analyzer

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

def analyze_csv(file_path):
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Check if "Change %" column exists
        if 'Change %' not in df.columns:
            return "No 'Change %' column found in the CSV file."
        
        # Convert dates to datetime and create day of week column
        df['Date'] = pd.to_datetime(df['Date'])
        df['Day_of_Week'] = df['Date'].dt.strftime('%A')  # Full day name
        
        # Filter out weekends (only keep Monday-Friday)
        df = df[df['Date'].dt.dayofweek < 5]  # 0=Monday, 4=Friday
        
        # Convert percentage strings to float numbers
        df['Change %'] = df['Change %'].str.rstrip('%').astype('float')
        # Remove commas from price columns and convert to float
        df['Price'] = df['Price'].str.replace(',', '').astype('float')
        df['High'] = df['High'].str.replace(',', '').astype('float')
        df['Low'] = df['Low'].str.replace(',', '').astype('float')
        
        total_count = len(df['Change %'].dropna())
        if total_count == 0:
            return "No valid data found in the Change % column."
        
        # Calculate positive and negative percentages
        positive_count = len(df[df['Change %'] > 0]['Change %'].dropna())
        negative_count = len(df[df['Change %'] < 0]['Change %'].dropna())
        neutral_count = len(df[df['Change %'] == 0]['Change %'].dropna())
        
        positive_percent = (positive_count / total_count) * 100
        negative_percent = (negative_count / total_count) * 100
        neutral_percent = (neutral_count / total_count) * 100
        
        # Calculate average positive and negative changes
        avg_positive_change = df[df['Change %'] > 0]['Change %'].mean()
        avg_negative_change = df[df['Change %'] < 0]['Change %'].mean()

        # Weekly Analysis
        # Create a week number for each date (Monday as start of week)
        df['Week_Number'] = df['Date'].dt.isocalendar().week
        df['Year'] = df['Date'].dt.isocalendar().year
        
        # Create unique week identifier (year + week number)
        df['Week_ID'] = df['Year'].astype(str) + '-' + df['Week_Number'].astype(str)
        
        # Group data by weeks
        weekly_groups = df.groupby('Week_ID')
        
        weekly_analysis = []
        
        for week_id, week_data in weekly_groups:
            week_data = week_data.sort_values('Date')  # Ensure data is sorted by date
            
            # Calculate weekly metrics
            week_positive_days = len(week_data[week_data['Change %'] > 0])
            week_negative_days = len(week_data[week_data['Change %'] < 0])
            week_neutral_days = len(week_data[week_data['Change %'] == 0])
            
            # Calculate average positive and negative changes for the week
            week_avg_positive = week_data[week_data['Change %'] > 0]['Change %'].mean() if len(week_data[week_data['Change %'] > 0]) > 0 else 0
            week_avg_negative = week_data[week_data['Change %'] < 0]['Change %'].mean() if len(week_data[week_data['Change %'] < 0]) > 0 else 0
            
            # Find best and worst days
            best_day_idx = week_data['Change %'].idxmax()
            worst_day_idx = week_data['Change %'].idxmin()
            best_day = {
                'date': week_data.loc[best_day_idx, 'Date'].strftime('%A, %d/%m/%Y'),
                'change': week_data.loc[best_day_idx, 'Change %']
            }
            worst_day = {
                'date': week_data.loc[worst_day_idx, 'Date'].strftime('%A, %d/%m/%Y'),
                'change': week_data.loc[worst_day_idx, 'Change %']
            }
            
            # Calculate weekly streak
            week_changes = week_data['Change %'].values
            week_dates = week_data['Date'].dt.strftime('%A, %d/%m/%Y').values
            current_week_streak = 1
            max_week_streak = 1
            week_streak_type = 'positive' if week_changes[0] > 0 else 'negative'
            week_streak_start = week_dates[0]
            week_streak_end = week_dates[0]
            current_week_start = week_dates[0]
            
            for i in range(1, len(week_changes)):
                if (week_changes[i] > 0 and week_changes[i-1] > 0) or (week_changes[i] < 0 and week_changes[i-1] < 0):
                    current_week_streak += 1
                    if current_week_streak > max_week_streak:
                        max_week_streak = current_week_streak
                        week_streak_type = 'positive' if week_changes[i] > 0 else 'negative'
                        week_streak_start = current_week_start
                        week_streak_end = week_dates[i]
                else:
                    current_week_streak = 1
                    current_week_start = week_dates[i]
            
            # Find highest and lowest points with specific days
            highest_idx = week_data['High'].idxmax()
            lowest_idx = week_data['Low'].idxmin()
            week_highest_day = week_data.loc[highest_idx, 'Date'].strftime('%A, %d/%m/%Y')
            week_lowest_day = week_data.loc[lowest_idx, 'Date'].strftime('%A, %d/%m/%Y')
            
            # Calculate daily range for most volatile day
            week_data['Abs_Change'] = week_data['Change %'].abs()
            most_volatile_idx = week_data['Abs_Change'].idxmax()
            most_volatile_date = week_data.loc[most_volatile_idx, 'Date'].strftime('%A, %d/%m/%Y')
            most_volatile_change = week_data.loc[most_volatile_idx, 'Change %']
            most_volatile_high = week_data.loc[most_volatile_idx, 'High']
            most_volatile_low = week_data.loc[most_volatile_idx, 'Low']
            most_volatile_range = most_volatile_high - most_volatile_low
            
            # Calculate weekly change by summing all daily changes
            week_change = week_data['Change %'].sum()
            
            # Generate daily breakdown
            daily_breakdown = ""
            for _, day in week_data.iterrows():
                daily_range_points = day['High'] - day['Low']
                daily_range_percent = (daily_range_points / day['Low']) * 100
                daily_breakdown += f"\n  {day['Date'].strftime('%A')}: Change: {round(day['Change %'], 2)}%, "
                daily_breakdown += f"Range: {round(daily_range_percent, 2)}% (${daily_range_points:,.2f}), "
                daily_breakdown += f"High: ${day['High']:,.2f}, Low: ${day['Low']:,.2f}"

            weekly_analysis.append({
                'start_date': week_data.iloc[0]['Date'].strftime('%d/%m/%Y'),
                'end_date': week_data.iloc[-1]['Date'].strftime('%d/%m/%Y'),
                'total_days': len(week_data),
                'positive_days': week_positive_days,
                'negative_days': week_negative_days,
                'neutral_days': week_neutral_days,
                'avg_positive': week_avg_positive,
                'avg_negative': week_avg_negative,
                'best_day': best_day,
                'worst_day': worst_day,
                'highest_point': {'price': week_data['High'].max(), 'date': week_highest_day},
                'lowest_point': {'price': week_data['Low'].min(), 'date': week_lowest_day},
                'most_volatile_day': most_volatile_date,
                'most_volatile_change': most_volatile_change,
                'most_volatile_range': most_volatile_range,
                'week_change': week_change,
                'max_streak': max_week_streak,
                'streak_type': week_streak_type,
                'streak_start': week_streak_start,
                'streak_end': week_streak_end,
                'daily_breakdown': daily_breakdown
            })

        # Calculate longest streak (keeping existing code)
        changes = df['Change %'].values
        dates = df['Date'].dt.strftime('%A, %d/%m/%Y').values
        current_streak = 1
        max_streak = 1
        streak_type = 'positive' if changes[0] > 0 else 'negative'
        streak_start_date = dates[0]
        streak_end_date = dates[0]
        current_start_date = dates[0]
        
        for i in range(1, len(changes)):
            if (changes[i] > 0 and changes[i-1] > 0) or (changes[i] < 0 and changes[i-1] < 0):
                current_streak += 1
                if current_streak > max_streak:
                    max_streak = current_streak
                    streak_type = 'positive' if changes[i] > 0 else 'negative'
                    streak_end_date = current_start_date
                    streak_start_date = dates[i]
            else:
                current_streak = 1
                current_start_date = dates[i]

        # Calculate pattern analysis
        patterns = {}
        
        # Day-of-week patterns
        # Group by day of week and calculate statistics
        day_of_week_stats = df.groupby('Day_of_Week')['Change %'].agg(['mean', 'count', lambda x: (x > 0).sum(), lambda x: (x < 0).sum()])
        day_of_week_stats.columns = ['avg_change', 'total_days', 'positive_days', 'negative_days']
        day_of_week_stats['positive_pct'] = (day_of_week_stats['positive_days'] / day_of_week_stats['total_days']) * 100
        day_of_week_stats['negative_pct'] = (day_of_week_stats['negative_days'] / day_of_week_stats['total_days']) * 100
        
        # Calculate average gain and loss for each day of the week
        # First, create a function to calculate average gain and loss
        def calc_avg_gain_loss(group):
            positive_changes = group[group > 0]
            negative_changes = group[group < 0]
            avg_gain = positive_changes.mean() if len(positive_changes) > 0 else 0
            avg_loss = negative_changes.mean() if len(negative_changes) > 0 else 0
            return pd.Series({'avg_gain': avg_gain, 'avg_loss': avg_loss})
        
        # Apply the function to each day of the week
        day_gain_loss = df.groupby('Day_of_Week')['Change %'].apply(calc_avg_gain_loss).unstack()
        day_of_week_stats = day_of_week_stats.join(day_gain_loss)
        
        # Reindex to get days in correct order
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
        day_of_week_stats = day_of_week_stats.reindex(day_order)
        
        # Calculate weekly changes and create a chronological series of weekly changes
        # First, get the sum of changes for each week
        weekly_changes_df = df.groupby(['Year', 'Week_Number'])['Change %'].sum().reset_index()
        
        # Sort by year and week to get chronological order
        weekly_changes_df = weekly_changes_df.sort_values(['Year', 'Week_Number'])
        
        # Get the list of weekly changes in chronological order
        weekly_change_list = weekly_changes_df['Change %'].values
        
        # Count consecutive positive and negative weeks
        consecutive_positive_weeks = 0
        consecutive_negative_weeks = 0
        current_positive_streak = 0
        current_negative_streak = 0
        
        for change in weekly_change_list:
            if change > 0:
                # Reset negative streak
                current_negative_streak = 0
                # Increment positive streak
                current_positive_streak += 1
                # If we have at least 2 consecutive positive weeks, count it
                if current_positive_streak >= 2:
                    consecutive_positive_weeks += 1
            elif change < 0:
                # Reset positive streak
                current_positive_streak = 0
                # Increment negative streak
                current_negative_streak += 1
                # If we have at least 2 consecutive negative weeks, count it
                if current_negative_streak >= 2:
                    consecutive_negative_weeks += 1
            else:  # change == 0
                # Reset both streaks for neutral weeks
                current_positive_streak = 0
                current_negative_streak = 0
        
        day_of_week_patterns = {
            'day_stats': day_of_week_stats,
            'consecutive_positive_weeks': consecutive_positive_weeks,
            'consecutive_negative_weeks': consecutive_negative_weeks
        }
        patterns['day_of_week'] = day_of_week_patterns
        
        # 3. Range patterns
        df['daily_range'] = df['High'] - df['Low']
        df['range_percent'] = (df['daily_range'] / df['Low']) * 100
        range_patterns = {
            'avg_daily_range': round(df['daily_range'].mean(), 2),
        }
        patterns['range'] = range_patterns
        
        # 5. Weekly patterns
        weekly_groups_patterns = df.groupby(['Year', 'Week_Number'])
        
        # Calculate true weekly ranges (highest high - lowest low for each week)
        weekly_highs = weekly_groups_patterns['High'].max()
        weekly_lows = weekly_groups_patterns['Low'].min()
        weekly_ranges = weekly_highs - weekly_lows
        
        # Calculate weekly total changes
        weekly_changes = weekly_groups_patterns['Change %'].sum()
        
        weekly_patterns = {
            'avg_weekly_range': round(weekly_ranges.mean(), 2),
            'strong_positive_weeks': len(weekly_changes[weekly_changes > 2.0]),  # Weeks with >2% positive change
            'weak_positive_weeks': len(weekly_changes[(weekly_changes > 0) & (weekly_changes < 2.0)]),  # Weeks with 0-2% positive change
            'strong_negative_weeks': len(weekly_changes[weekly_changes < -2.0]),  # Weeks with <-2% negative change
            'weak_negative_weeks': len(weekly_changes[(weekly_changes < 0) & (weekly_changes > -2.0)])  # Weeks with 0 to -2% negative change
        }
        patterns['weekly'] = weekly_patterns

        # Sort weekly analysis from newest to oldest using the end_date
        weekly_analysis.sort(key=lambda x: pd.to_datetime(x['end_date'], format='%d/%m/%Y'), reverse=True)

        # Generate the output string for overall results
        output = f"""
Analysis Results ({df['Date'].min().strftime('%d/%m/%Y')} - {df['Date'].max().strftime('%d/%m/%Y')} | {total_count} Trading Days):
==================================================
Overall Distribution:
  Positive Days: {round(positive_percent, 2)}% ({positive_count} days)
  Negative Days: {round(negative_percent, 2)}% ({negative_count} days)
  Neutral Days:  {round(neutral_percent, 2)}% ({neutral_count} days)

Average Changes:
  Average Positive Change: {round(avg_positive_change, 2) if not pd.isna(avg_positive_change) else 'N/A'}%
  Average Negative Change: {round(avg_negative_change, 2) if not pd.isna(avg_negative_change) else 'N/A'}%

Longest Overall Streak:
  {max_streak} days ({streak_type}) from {streak_start_date} to {streak_end_date}

Overall High/Low:
  Highest Price: {df.loc[df['High'].idxmax(), 'High']} on {df.loc[df['High'].idxmax(), 'Date'].strftime('%A, %d/%m/%Y')}
  Lowest Price:  {df.loc[df['Low'].idxmin(), 'Low']} on {df.loc[df['Low'].idxmin(), 'Date'].strftime('%A, %d/%m/%Y')}

Pattern Analysis:
=================
Day-of-Week Patterns:
  Monday:    Avg Change: {round(patterns['day_of_week']['day_stats'].loc['Monday', 'avg_change'], 2)}%  Positive: {round(patterns['day_of_week']['day_stats'].loc['Monday', 'positive_pct'], 1)}%  Negative: {round(patterns['day_of_week']['day_stats'].loc['Monday', 'negative_pct'], 1)}%  Avg Gain: {round(patterns['day_of_week']['day_stats'].loc['Monday', 'avg_gain'], 2)}%  Avg Loss: {round(patterns['day_of_week']['day_stats'].loc['Monday', 'avg_loss'], 2)}%
  Tuesday:   Avg Change: {round(patterns['day_of_week']['day_stats'].loc['Tuesday', 'avg_change'], 2)}%  Positive: {round(patterns['day_of_week']['day_stats'].loc['Tuesday', 'positive_pct'], 1)}%  Negative: {round(patterns['day_of_week']['day_stats'].loc['Tuesday', 'negative_pct'], 1)}%  Avg Gain: {round(patterns['day_of_week']['day_stats'].loc['Tuesday', 'avg_gain'], 2)}%  Avg Loss: {round(patterns['day_of_week']['day_stats'].loc['Tuesday', 'avg_loss'], 2)}%
  Wednesday: Avg Change: {round(patterns['day_of_week']['day_stats'].loc['Wednesday', 'avg_change'], 2)}%  Positive: {round(patterns['day_of_week']['day_stats'].loc['Wednesday', 'positive_pct'], 1)}%  Negative: {round(patterns['day_of_week']['day_stats'].loc['Wednesday', 'negative_pct'], 1)}%  Avg Gain: {round(patterns['day_of_week']['day_stats'].loc['Wednesday', 'avg_gain'], 2)}%  Avg Loss: {round(patterns['day_of_week']['day_stats'].loc['Wednesday', 'avg_loss'], 2)}%
  Thursday:  Avg Change: {round(patterns['day_of_week']['day_stats'].loc['Thursday', 'avg_change'], 2)}%  Positive: {round(patterns['day_of_week']['day_stats'].loc['Thursday', 'positive_pct'], 1)}%  Negative: {round(patterns['day_of_week']['day_stats'].loc['Thursday', 'negative_pct'], 1)}%  Avg Gain: {round(patterns['day_of_week']['day_stats'].loc['Thursday', 'avg_gain'], 2)}%  Avg Loss: {round(patterns['day_of_week']['day_stats'].loc['Thursday', 'avg_loss'], 2)}%
  Friday:    Avg Change: {round(patterns['day_of_week']['day_stats'].loc['Friday', 'avg_change'], 2)}%  Positive: {round(patterns['day_of_week']['day_stats'].loc['Friday', 'positive_pct'], 1)}%  Negative: {round(patterns['day_of_week']['day_stats'].loc['Friday', 'negative_pct'], 1)}%  Avg Gain: {round(patterns['day_of_week']['day_stats'].loc['Friday', 'avg_gain'], 2)}%  Avg Loss: {round(patterns['day_of_week']['day_stats'].loc['Friday', 'avg_loss'], 2)}%
  Consecutive Positive Weeks: {patterns['day_of_week']['consecutive_positive_weeks']}
  Consecutive Negative Weeks: {patterns['day_of_week']['consecutive_negative_weeks']}

Range Patterns:
  Average Daily Range: ${patterns['range']['avg_daily_range']}

Weekly Patterns:
  Average Weekly Range: ${patterns['weekly']['avg_weekly_range']}
  Strong Positive Weeks (>2% change): {patterns['weekly']['strong_positive_weeks']}
  Weak Positive Weeks (<2% change): {patterns['weekly']['weak_positive_weeks']}
  Strong Negative Weeks (<-2% change): {patterns['weekly']['strong_negative_weeks']}
  Weak Negative Weeks (>-2% change): {patterns['weekly']['weak_negative_weeks']}

Weekly Breakdown:
================"""

        # Append weekly breakdown details
        for week_info in weekly_analysis:
            output += f"""

Week: {week_info['start_date']} - {week_info['end_date']} ({week_info['total_days']} Trading Days)
--------------------------------------
  Distribution:
    Positive Days: {week_info['positive_days']}
    Negative Days: {week_info['negative_days']}
    Neutral Days:  {week_info['neutral_days']}
  Average Changes:
    Avg Positive: {round(week_info['avg_positive'], 2) if not pd.isna(week_info['avg_positive']) else 'N/A'}%
    Avg Negative: {round(week_info['avg_negative'], 2) if not pd.isna(week_info['avg_negative']) else 'N/A'}%
  Best/Worst Days:
    Best:  {round(week_info['best_day']['change'], 2)}% on {week_info['best_day']['date']}
    Worst: {round(week_info['worst_day']['change'], 2)}% on {week_info['worst_day']['date']}
  High/Low Points:
    Highest: {week_info['highest_point']['price']} on {week_info['highest_point']['date']}
    Lowest:  {week_info['lowest_point']['price']} on {week_info['lowest_point']['date']}
  Longest Streak:
    {week_info['max_streak']} days ({week_info['streak_type']}) from {week_info['streak_start']} to {week_info['streak_end']}
  Most Volatile Day:
    {week_info['most_volatile_day']} with a change of {round(week_info['most_volatile_change'], 2)}% and a range of ${week_info['most_volatile_range']:,.2f}
  Weekly Change:
    {round(week_info['week_change'], 2)}%
  Daily Breakdown:
{week_info['daily_breakdown']}
"""

        return output

    except FileNotFoundError:
        return "Error: File not found."

@app.route('/')
def index():
    return '''
    <!doctype html>
    <title>Upload CSV File</title>
    <h1>Upload CSV File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''

@app.route('/', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        # Call the analysis function here
        result = analyze_csv(file_path)
        return f'''
        <h1>Analysis Results</h1>
        <pre>{result}</pre>
        <p><a href="/patterns/{file.filename}">View Advanced Pattern Analysis</a></p>
        <p><a href="/">Upload another file</a></p>
        '''

@app.route('/patterns/<filename>')
def pattern_analysis(filename):
    """Display advanced pattern analysis for the uploaded file."""
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    if not os.path.exists(file_path):
        return "File not found. Please upload the file again."
    
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Ensure required columns exist
        if 'Date' not in df.columns or 'Change %' not in df.columns:
            return "Required columns (Date, Change %) not found in the CSV file."
        
        # Convert dates to datetime and create day of week column
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Convert percentage strings to float numbers if needed
        if isinstance(df['Change %'].iloc[0], str):
            df['Change %'] = df['Change %'].str.rstrip('%').astype('float')
        
        # Convert price columns to float if they exist and are strings
        for col in ['Price', 'High', 'Low']:
            if col in df.columns and isinstance(df[col].iloc[0], str):
                df[col] = df[col].str.replace(',', '').astype('float')
        
        # Run the pattern analysis
        pattern_results = pattern_analyzer.analyze_patterns(df)
        
        # Format the results
        formatted_results = pattern_analyzer.format_pattern_results(pattern_results)
        
        return f'''
        <!doctype html>
        <html>
        <head>
            <title>Advanced Pattern Analysis</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
                pre {{ background-color: #f5f5f5; padding: 15px; border-radius: 5px; overflow-x: auto; }}
                h1, h2 {{ color: #333; }}
                a {{ color: #0066cc; text-decoration: none; }}
                a:hover {{ text-decoration: underline; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Advanced Pattern Analysis for {filename}</h1>
                <p><a href="/">← Back to Upload</a></p>
                <pre>{formatted_results}</pre>
            </div>
        </body>
        </html>
        '''
    
    except Exception as e:
        return f"Error analyzing patterns: {str(e)}"

if __name__ == '__main__':
    # Create uploads directory if it doesn't exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
