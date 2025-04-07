import pandas as pd
import numpy as np

def analyze_index_base_pattern(primary_df, index_base_df):
    """
    Analyze the relationship between a primary CSV file and an index base CSV file.
    
    Parameters:
    primary_df (pandas.DataFrame): DataFrame containing the primary financial data
    index_base_df (pandas.DataFrame): DataFrame containing the index base pattern data
    
    Returns:
    dict: Dictionary containing the analysis results
    """
    # Initialize results dictionary
    results = {}
    # Add a new field to store price percentages
    results['price_percentages'] = []
    
    # Ensure required columns exist in both dataframes
    required_columns = ['Open', 'Low', 'High', 'Price']
    
    # Check primary_df for Open column
    if 'Open' not in primary_df.columns:
        return {"error": "Required column 'Open' not found in primary data"}
    
    # Check index_base_df for required columns
    missing_columns = [col for col in required_columns if col not in index_base_df.columns]
    if missing_columns:
        return {"error": f"Required columns {', '.join(missing_columns)} not found in index base data"}
    
    # Extract values from index_base_df
    results['open'] = index_base_df['Open'].values.tolist()
    results['low'] = index_base_df['Low'].values.tolist()
    results['high'] = index_base_df['High'].values.tolist()
    results['price'] = index_base_df['Price'].values.tolist()
    
    # Get the primary open values for each date
    results['primary_open'] = []
    
    # Check if dates are available in both dataframes
    if 'Date' in primary_df.columns and 'Date' in index_base_df.columns:
        # For each date in index_base_df, find the corresponding primary open value
        for date in index_base_df['Date']:
            # Find matching date in primary_df
            matching_row = primary_df[primary_df['Date'] == date]
            if not matching_row.empty:
                results['primary_open'].append(matching_row['Open'].values[0])
            else:
                # If no matching date found, use None
                results['primary_open'].append(None)
    else:
        # If dates are not available, use the first open value for all
        results['primary_open'] = [primary_df['Open'].values[0]] * len(index_base_df)
    
    # Calculate since low and since high values
    results['since_low'] = []
    results['since_high'] = []
    results['alt_opens'] = []
    
    for i, row in index_base_df.iterrows():
        # Calculate percentage change from low to price
        if row['Low'] != 0:  # Avoid division by zero
            since_low = ((row['Price'] - row['Low']) / row['Low']) * 100
            results['since_low'].append(round(since_low, 2))
        else:
            results['since_low'].append(0)
        
        # Calculate percentage change from high to price
        if row['High'] != 0:  # Avoid division by zero
            since_high = ((row['Price'] - row['High']) / row['High']) * 100
            results['since_high'].append(round(since_high, 2))
        else:
            results['since_high'].append(0)
            
        # Calculate percentage difference between primary open and index base open
        primary_open = results['primary_open'][i] if i < len(results['primary_open']) and results['primary_open'][i] is not None else None
        if primary_open is not None and row['Open'] != 0:  # Avoid division by zero
            # Extract integer parts (only count absolute value until the dot)
            primary_open_int = int(primary_open)  # Remove abs() to keep sign
            index_open_int = int(row['Open'])  # Remove abs() to keep sign
            
            # Calculate the difference as a percentage of the index base value
            # Following the specific percentage calculation pattern from the example
            if index_open_int > 0:
                # If primary_open is 41363 and index_open is 41097, result should be -0.39%
                # This means we need to calculate: (index_open - primary_open) / index_open * 100
                diff_percentage = (index_open_int - primary_open_int) / index_open_int * 100
                # Round to 2 decimal places
                results['alt_opens'].append(round(diff_percentage, 2))
            else:
                results['alt_opens'].append(None)
        else:
            results['alt_opens'].append(None)
    
    # Add dates if available
    if 'Date' in index_base_df.columns:
        results['dates'] = index_base_df['Date'].values.tolist()
    
    # Calculate price percentages relative to open values
    for i in range(len(results['price'])):
        if i < len(results['open']) and results['open'][i] != 0:
            # Calculate percentage change from open to price
            price_percentage = ((results['price'][i] - results['open'][i]) / results['open'][i]) * 100
            results['price_percentages'].append(round(price_percentage, 2))
        else:
            results['price_percentages'].append(None)
    
    return results

def format_index_base_results(results):
    """
    Format the index base pattern analysis results into a spreadsheet-like HTML format.
    
    Parameters:
    results (dict): The results from analyze_index_base_pattern
    
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
    <h2>Index Base Pattern Analysis</h2>
    """
    
    # Create a table for the results
    output += "<table class='index-base-table'>\n"
    
    # Table headers
    output += "<thead>\n<tr>\n"
    headers = ["Date", "Primary Open", "Open", "Low", "High", "Since Low (%)", "Since High (%)", "% alt. opens", "Price"]
    for header in headers:
        output += f"<th>{header}</th>\n"
    output += "</tr>\n</thead>\n"
    
    # Table body
    output += "<tbody>\n"
    
    # Determine number of rows
    num_rows = len(results.get('open', []))
    
    for i in range(num_rows):
        output += "<tr>\n"
        
        # Date column
        if 'dates' in results and i < len(results['dates']):
            output += f"<td>{results['dates'][i]}</td>\n"
        else:
            output += f"<td>Row {i+1}</td>\n"
        
        # Primary Open column
        if 'primary_open' in results and i < len(results['primary_open']) and results['primary_open'][i] is not None:
            output += f"<td>{results['primary_open'][i]}</td>\n"
        else:
            output += "<td>-</td>\n"
        
        # Open column
        if i < len(results['open']):
            output += f"<td>{results['open'][i]}</td>\n"
        else:
            output += "<td>-</td>\n"
        
        # Low column
        if i < len(results['low']):
            output += f"<td>{results['low'][i]}</td>\n"
        else:
            output += "<td>-</td>\n"
        
        # High column
        if i < len(results['high']):
            output += f"<td>{results['high'][i]}</td>\n"
        else:
            output += "<td>-</td>\n"
        
        # Since Low column
        if i < len(results['since_low']):
            change = results['since_low'][i]
            color_class = "positive" if change >= 0 else "negative"
            output += f"<td class='{color_class}'>{change}%</td>\n"
        else:
            output += "<td>-</td>\n"
        
        # Since High column
        if i < len(results['since_high']):
            change = results['since_high'][i]
            color_class = "positive" if change >= 0 else "negative"
            output += f"<td class='{color_class}'>{change}%</td>\n"
        else:
            output += "<td>-</td>\n"
        
        # % alt. opens column
        if i < len(results['alt_opens']) and results['alt_opens'][i] is not None:
            change = results['alt_opens'][i]
            color_class = "positive" if change >= 0 else "negative"
            output += f"<td class='{color_class}'>{change}%</td>\n"
        else:
            output += "<td>-</td>\n"
        
        # Price column with percentage
        if i < len(results['price']):
            price_value = results['price'][i]
            
            # Add percentage if available
            if 'price_percentages' in results and i < len(results['price_percentages']) and results['price_percentages'][i] is not None:
                percentage = results['price_percentages'][i]
                color_class = "positive" if percentage >= 0 else "negative"
                output += f"<td>{price_value} <span class='{color_class}'>({percentage}%)</span></td>\n"
            else:
                output += f"<td>{price_value}</td>\n"
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
        .index-base-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
            font-family: Arial, sans-serif;
        }
        .index-base-table th, .index-base-table td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: right;
        }
        .index-base-table th {
            background-color: #f2f2f2;
            font-weight: bold;
            text-align: center;
        }
        .index-base-table tr:nth-child(even) {
            background-color: #f9f9f9;
        }
        .index-base-table tr:hover {
            background-color: #f5f5f5;
        }
        .positive {
            color: green;
        }
        .negative {
            color: red;
        }
        .error {
            color: red;
            font-weight: bold;
        }
    </style>
    """
    
    return output