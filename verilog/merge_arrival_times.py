import os
import json
import pandas as pd
from glob import glob

# Function to read JSON files and convert to DataFrame (copied from notebook)
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
    json_files = glob(os.path.join(folder_path, "*.json"))
    
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

# Read the JSON files into a DataFrame
df = json_files_to_dataframe("verilog/reports")

# Extract the columns of interest
df2 = df[["finish__design__instance__area", "finish__power__total", "finish__timing__setup__ws"]]

# Rename columns for clarity
df2["area"] = df2["finish__design__instance__area"]
df2["power"] = df2["finish__power__total"]
df2["ws"] = df2["finish__timing__setup__ws"]
df2.drop(columns=["finish__design__instance__area", "finish__power__total", "finish__timing__setup__ws"], inplace=True)

# Read the arrival_times CSV file
arrival_times_df = pd.read_csv("verilog/arrival_times.csv")

# Transform the filenames to match the DataFrame indices
# Convert 'adder_combinational_fp32_finish.rpt' to 'adder_combinational_fp32_report'
arrival_times_df['index_name'] = arrival_times_df['filename'].str.replace('_finish.rpt', '_report')

# Set the transformed filename as the index
arrival_times_df.set_index('index_name', inplace=True)

# Extract only the max_arrival_time column
arrival_times_subset = arrival_times_df[['max_arrival_time']]

# Merge with the existing DataFrame
df_merged = pd.merge(df2, arrival_times_subset, left_index=True, right_index=True, how='left')

# Print some statistics
print(f"Original DataFrame rows: {len(df2)}")
print(f"Merged DataFrame rows with arrival times: {df_merged['max_arrival_time'].notna().sum()}")
print(f"Percentage of rows with arrival times: {df_merged['max_arrival_time'].notna().sum() / len(df2) * 100:.2f}%")

# Save the merged DataFrame to a CSV file
df_merged.to_csv("verilog/merged_data.csv")

print("Merged data saved to verilog/merged_data.csv")
print("\nSample of merged data:")
print(df_merged.head()) 