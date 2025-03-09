#!/usr/bin/env python3
"""
Comprehensive hardware analysis script that:
1. Fetches JSON reports from WSL
2. Fetches finish reports from WSL
3. Extracts max arrival times from finish reports
4. Processes JSON reports to get area and power data
5. Merges all data and saves as analysis_data.csv
"""

import os
import re
import sys
import json
import shutil
import pandas as pd
import glob as glob_module
from pathlib import Path
from data_transfer import (
    SOURCE_DIR, REPORTS_DIR, FINISH_REPORTS_DIR, 
    WSL_LOGS_PATH, WSL_REPORTS_PATH, ensure_directory
)

def fetch_json_reports():
    """
    Fetch JSON reports from WSL and save them to the reports directory.
    Returns the number of reports copied.
    """
    print("\n=== Fetching JSON Reports ===")
    
    # Check if source directory exists
    if not os.path.exists(SOURCE_DIR):
        print(f"Error: Source directory '{SOURCE_DIR}' does not exist.")
        print("Please run the organize_verilog.py script first to create the 'all' directory.")
        return 0
    
    # Check if WSL logs path exists and is accessible
    if not os.path.exists(WSL_LOGS_PATH):
        print(f"Error: WSL logs path '{WSL_LOGS_PATH}' does not exist or is not accessible.")
        print("Please make sure the WSL path is correct and accessible from Windows.")
        return 0
    
    # Create reports directory if it doesn't exist
    ensure_directory(REPORTS_DIR)
    
    # Get list of subdirectories in the "all" folder (these are our designs)
    designs = [d for d in os.listdir(SOURCE_DIR) if os.path.isdir(os.path.join(SOURCE_DIR, d))]
    
    if not designs:
        print("No designs found in the 'all' directory.")
        return 0
    
    # Copy and rename report.json for each design
    copied_count = 0
    for design in designs:
        # Source path for the report.json in WSL
        report_source_path = os.path.join(WSL_LOGS_PATH, design, "base", "6_report.json")
        
        # Destination path with renamed file
        report_dest_path = os.path.join(REPORTS_DIR, f"{design}_report.json")
        
        # Check if source report exists
        if os.path.exists(report_source_path):
            print(f"Copying report for {design}...")
            try:
                shutil.copy2(report_source_path, report_dest_path)
                copied_count += 1
                print(f"  Saved to: {report_dest_path}")
            except Exception as e:
                print(f"  Error copying report for {design}: {str(e)}")
        else:
            print(f"Report not found for {design}: {report_source_path}")
    
    print(f"\nCopied {copied_count} out of {len(designs)} reports to {REPORTS_DIR}")
    return copied_count

def fetch_finish_reports():
    """
    Fetch finish reports from WSL and save them to the finish_reports directory.
    Returns the number of reports copied.
    """
    print("\n=== Fetching Finish Reports ===")
    
    # Check if source directory exists
    if not os.path.exists(SOURCE_DIR):
        print(f"Error: Source directory '{SOURCE_DIR}' does not exist.")
        print("Please run the organize_verilog.py script first to create the 'all' directory.")
        return 0
    
    # Check if WSL reports path exists and is accessible
    if not os.path.exists(WSL_REPORTS_PATH):
        print(f"Error: WSL reports path '{WSL_REPORTS_PATH}' does not exist or is not accessible.")
        print("Please make sure the WSL path is correct and accessible from Windows.")
        return 0
    
    # Create finish reports directory if it doesn't exist
    ensure_directory(FINISH_REPORTS_DIR)
    
    # Get list of subdirectories in the "all" folder (these are our designs)
    designs = [d for d in os.listdir(SOURCE_DIR) if os.path.isdir(os.path.join(SOURCE_DIR, d))]
    
    if not designs:
        print("No designs found in the 'all' directory.")
        return 0
    
    # Copy and rename finish report for each design
    copied_count = 0
    for design in designs:
        # Source path for the finish report in WSL
        report_source_path = os.path.join(WSL_REPORTS_PATH, design, "base", "6_finish.rpt")
        
        # Destination path with renamed file
        report_dest_path = os.path.join(FINISH_REPORTS_DIR, f"{design}_finish.rpt")
        
        # Check if source report exists
        if os.path.exists(report_source_path):
            print(f"Copying finish report for {design}...")
            try:
                shutil.copy2(report_source_path, report_dest_path)
                copied_count += 1
                print(f"  Saved to: {report_dest_path}")
            except Exception as e:
                print(f"  Error copying finish report for {design}: {str(e)}")
        else:
            print(f"Finish report not found for {design}: {report_source_path}")
    
    print(f"\nCopied {copied_count} out of {len(designs)} finish reports to {FINISH_REPORTS_DIR}")
    return copied_count

def extract_arrival_time(file_path):
    """
    Extract the data arrival time from a finish report file.
    Returns a list of all found arrival times as floats.
    """
    arrival_times = []
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            
            # Look for data arrival time patterns
            # Pattern 1: standalone "data arrival time" line
            pattern1 = r'(\d+\.\d+)\s+data arrival time'
            matches1 = re.findall(pattern1, content)
            arrival_times.extend([float(t) for t in matches1])
            
            # Pattern 2: "data arrival time" at the end of a timing path
            pattern2 = r'^\s+(\d+\.\d+)\s+data arrival time$'
            matches2 = re.findall(pattern2, content, re.MULTILINE)
            arrival_times.extend([float(t) for t in matches2])
            
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
    
    return arrival_times

def find_max_arrival_times():
    """
    Find maximum arrival times from finish reports.
    Returns a DataFrame with max arrival times.
    """
    print("\n=== Finding Maximum Arrival Times ===")
    
    # Find all .rpt files
    report_files = glob_module.glob(os.path.join(FINISH_REPORTS_DIR, '*.rpt'))
    
    if not report_files:
        print(f"No report files found in {FINISH_REPORTS_DIR}")
        return pd.DataFrame()
    
    # List to store data for DataFrame
    data = []
    
    # Process each file
    for file_path in report_files:
        file_name = os.path.basename(file_path)
        arrival_times = extract_arrival_time(file_path)
        
        if arrival_times:
            # Find the maximum absolute value
            max_abs_time = max(abs(t) for t in arrival_times)
            
            # Add to data list
            data.append({
                'filename': file_name,
                'max_arrival_time': max_abs_time
            })
        else:
            print(f"No arrival times found in {file_name}")
    
    # Create DataFrame
    if data:
        df = pd.DataFrame(data)
        
        # Sort by max arrival time (descending)
        df_sorted = df.sort_values('max_arrival_time', ascending=False)
        
        # Display results
        print("\nResults:")
        print(f"Maximum absolute data arrival time: {df_sorted['max_arrival_time'].max():.6f}")
        print(f"Found in file: {df_sorted.iloc[0]['filename']}")
        
        # Print top 5 files with highest arrival times
        print("\nTop 5 files with highest arrival times:")
        for i, (_, row) in enumerate(df_sorted.head(5).iterrows(), 1):
            print(f"{i}. {row['filename']}: {row['max_arrival_time']:.6f}")
        
        # Transform the filenames to match the JSON report indices
        # Convert 'adder_combinational_fp32_finish.rpt' to 'adder_combinational_fp32_report'
        df_sorted['index_name'] = df_sorted['filename'].str.replace('_finish.rpt', '_report')
        
        # Set the transformed filename as the index
        df_sorted.set_index('index_name', inplace=True)
        
        return df_sorted
    else:
        print("No arrival times found in any file")
        return pd.DataFrame()

def json_files_to_dataframe(folder_path):
    """
    Reads all JSON files in a folder and converts them to a single pandas DataFrame,
    using filenames as the DataFrame indices.
    
    Parameters:
    folder_path (str): Path to the folder containing JSON files
    
    Returns:
    pandas.DataFrame: Combined DataFrame from all JSON files with filenames as indices
    """
    # Get all JSON files in the folder
    json_files = glob_module.glob(os.path.join(folder_path, "*.json"))
    
    if not json_files:
        raise ValueError(f"No JSON files found in {folder_path}")
    
    # Dictionary to store data from each file, with filenames as keys
    data_dict = {}
    
    # Process each JSON file
    for file_path in json_files:
        # Extract filename as the index
        file_id = os.path.basename(file_path).replace('.json', '')
        
        # Read the JSON file
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Store data with filename as key
        data_dict[file_id] = data
    
    # Convert dictionary to DataFrame, using keys as indices
    df = pd.DataFrame.from_dict(data_dict, orient='index')
    
    return df

def process_json_reports():
    """
    Process JSON reports to extract area and power data.
    Returns a DataFrame with the extracted data.
    """
    print("\n=== Processing JSON Reports ===")
    
    try:
        # Read JSON files into DataFrame
        df = json_files_to_dataframe(REPORTS_DIR)
        
        # Extract relevant columns
        df_subset = df[["finish__design__instance__area", "finish__power__total"]]
        
        # Rename columns for clarity
        df_subset.rename(columns={
            "finish__design__instance__area": "area",
            "finish__power__total": "power",
        }, inplace=True)
        
        # Convert power to mW for better readability
        df_subset["power"] = df_subset["power"] * 1000  # Convert to mW
        
        print(f"Processed {len(df_subset)} JSON reports")
        return df_subset
        
    except Exception as e:
        print(f"Error processing JSON reports: {e}")
        return pd.DataFrame()

def main():
    """
    Main function that orchestrates the entire analysis process.
    """
    print("=== Hardware Analysis Script ===")
    
    # Step 1: Fetch JSON reports
    json_count = fetch_json_reports()
    
    # Step 2: Fetch finish reports
    finish_count = fetch_finish_reports()
    
    if json_count == 0 or finish_count == 0:
        print("Error: Failed to fetch required reports. Exiting.")
        return 1
    
    # Step 3: Find max arrival times
    arrival_times_df = find_max_arrival_times()
    
    if arrival_times_df.empty:
        print("Error: Failed to extract arrival times. Exiting.")
        return 1
    
    # Step 4: Process JSON reports
    json_data_df = process_json_reports()
    
    if json_data_df.empty:
        print("Error: Failed to process JSON reports. Exiting.")
        return 1
    
    # Step 5: Merge data
    print("\n=== Merging Data ===")
    
    # Extract only the max_arrival_time column from arrival times DataFrame
    arrival_times_subset = arrival_times_df[['max_arrival_time']]
    arrival_times_subset.rename(columns={'max_arrival_time': 'ws'}, inplace=True)
    
    # Merge with the JSON data DataFrame
    merged_df = pd.merge(
        json_data_df, 
        arrival_times_subset, 
        left_index=True, 
        right_index=True, 
        how='left'
    )
    
    # Clean up the index by removing '_report' suffix
    merged_df.index = merged_df.index.str.replace('_report', '')
    
    # Save the final merged data
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'analysis_data.csv')
    merged_df.to_csv(output_path)
    
    print(f"\nAnalysis complete! Data saved to: {output_path}")
    print(f"Total designs analyzed: {len(merged_df)}")
    
    # Display a summary of the data
    print("\nData Summary:")
    print(merged_df.describe())
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 