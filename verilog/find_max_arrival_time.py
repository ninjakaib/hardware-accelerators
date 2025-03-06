#!/usr/bin/env python3
import os
import re
import glob
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

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

def parse_design_info(filename):
    """
    Parse design information from the filename.
    Returns a dictionary with design attributes.
    """
    # Remove the _finish.rpt suffix
    base_name = filename.replace('_finish.rpt', '')
    
    # Extract components
    components = base_name.split('_')
    
    # Initialize design info with defaults
    design_info = {
        'design_type': components[0] if components else '',
        'architecture': 'unknown',
        'precision': 'unknown',
        'stage': 'none'
    }
    
    # Extract architecture (combinational, pipelined, etc.)
    if 'combinational' in components:
        design_info['architecture'] = 'combinational'
    elif 'pipelined' in components:
        design_info['architecture'] = 'pipelined'
    elif any(c.startswith('stage_') for c in components):
        design_info['architecture'] = 'staged'
        # Extract stage number if present
        for comp in components:
            if comp.startswith('stage_'):
                design_info['stage'] = comp
                break
    
    # Extract precision (fp32, fp8, bf16)
    if 'fp32' in components:
        design_info['precision'] = 'fp32'
    elif 'fp8' in components:
        design_info['precision'] = 'fp8'
    elif 'bf16' in components:
        design_info['precision'] = 'bf16'
    
    # Check if it's a fast implementation
    design_info['is_fast'] = 'fast' in components
    
    return design_info

def main():
    # Path to the finish_reports directory
    reports_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'finish_reports')
    
    # Find all .rpt files
    report_files = glob.glob(os.path.join(reports_dir, '*.rpt'))
    
    if not report_files:
        print(f"No report files found in {reports_dir}")
        return
    
    # List to store data for DataFrame
    data = []
    
    # Process each file
    for file_path in report_files:
        file_name = os.path.basename(file_path)
        arrival_times = extract_arrival_time(file_path)
        
        if arrival_times:
            # Find the maximum absolute value
            max_abs_time = max(abs(t) for t in arrival_times)
            
            # Parse design information from filename
            design_info = parse_design_info(file_name)
            
            # Add to data list
            data.append({
                'filename': file_name,
                'max_arrival_time': max_abs_time,
                'design_type': design_info['design_type'],
                'architecture': design_info['architecture'],
                'precision': design_info['precision'],
                'stage': design_info['stage'],
                'is_fast': design_info['is_fast']
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
        
        # Save DataFrame to CSV
        output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'arrival_times.csv')
        df_sorted.to_csv(output_path, index=False)
        print(f"\nFull results saved to: {output_path}")
        
        # Generate some basic statistics
        print("\nStatistics by design type:")
        print(df.groupby('design_type')['max_arrival_time'].agg(['mean', 'min', 'max']))
        
        print("\nStatistics by architecture:")
        print(df.groupby('architecture')['max_arrival_time'].agg(['mean', 'min', 'max']))
        
        print("\nStatistics by precision:")
        print(df.groupby('precision')['max_arrival_time'].agg(['mean', 'min', 'max']))
        
        # Optional: Create a simple plot
        try:
            plt.figure(figsize=(12, 6))
            
            # Group by design type and architecture
            grouped = df.groupby(['design_type', 'architecture'])['max_arrival_time'].max().unstack()
            grouped.plot(kind='bar')
            
            plt.title('Maximum Arrival Time by Design Type and Architecture')
            plt.ylabel('Max Arrival Time (ns)')
            plt.tight_layout()
            
            # Save the plot
            plot_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'arrival_times_plot.png')
            plt.savefig(plot_path)
            print(f"\nPlot saved to: {plot_path}")
        except Exception as e:
            print(f"Could not generate plot: {e}")
        
        return df_sorted
    else:
        print("No arrival times found in any file")
        return None

if __name__ == "__main__":
    df = main() 