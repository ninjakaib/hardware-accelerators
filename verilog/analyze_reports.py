#!/usr/bin/env python3
import os
import json
import pandas as pd
import glob
import matplotlib.pyplot as plt
import sys

def main():
    # Script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Reports directory (where the JSON files are stored)
    reports_dir = os.path.join(script_dir, "reports")
    
    # Check if reports directory exists
    if not os.path.exists(reports_dir):
        print(f"Error: Reports directory '{reports_dir}' does not exist.")
        print("Please run the fetch_reports.py script first to fetch the report files.")
        return 1
    
    # Find all report JSON files
    report_files = glob.glob(os.path.join(reports_dir, "*_report.json"))
    
    if not report_files:
        print(f"No report files found in '{reports_dir}'.")
        print("Please run the fetch_reports.py script first to fetch the report files.")
        return 1
    
    print(f"Found {len(report_files)} report files.")
    
    # List to store data from each report
    data_list = []
    
    # Process each report file
    for report_file in report_files:
        try:
            # Extract design name from filename
            filename = os.path.basename(report_file)
            design_name = filename.replace("_report.json", "")
            
            # Load JSON data
            with open(report_file, 'r') as f:
                report_data = json.load(f)
            
            # Extract relevant metrics
            metrics = {
                'design': design_name,
                'runtime_total_seconds': report_data.get('runtime', {}).get('total', None),
            }
            
            # Extract design metrics if available
            if 'design' in report_data:
                design_metrics = report_data['design']
                metrics.update({
                    'area': design_metrics.get('area', None),
                    'cell_count': design_metrics.get('cell_count', None),
                    'utilization': design_metrics.get('utilization', None),
                })
            
            # Extract timing metrics if available
            if 'timing' in report_data:
                timing = report_data['timing']
                metrics.update({
                    'wns': timing.get('wns', None),  # Worst Negative Slack
                    'tns': timing.get('tns', None),  # Total Negative Slack
                    'clock_period': timing.get('clock_period', None),
                })
            
            # Extract power metrics if available
            if 'power' in report_data:
                power = report_data['power']
                metrics.update({
                    'total_power': power.get('total', None),
                    'leakage_power': power.get('leakage', None),
                    'dynamic_power': power.get('dynamic', None),
                })
            
            # Add to data list
            data_list.append(metrics)
            print(f"Processed: {design_name}")
            
        except Exception as e:
            print(f"Error processing {report_file}: {str(e)}")
    
    if not data_list:
        print("No data could be extracted from the report files.")
        return 1
    
    # Create DataFrame
    df = pd.DataFrame(data_list)
    
    # Save to CSV for easy viewing
    csv_path = os.path.join(reports_dir, "design_metrics.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved metrics to: {csv_path}")
    
    # Display basic statistics
    print("\nBasic Statistics:")
    print(df.describe())
    
    # Sort by area and display top 10
    print("\nTop 10 designs by area:")
    if 'area' in df.columns:
        print(df.sort_values('area').head(10)[['design', 'area', 'cell_count']])
    
    # Create some basic visualizations
    try:
        # Create output directory for plots
        plots_dir = os.path.join(reports_dir, "plots")
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)
        
        # Area vs Cell Count scatter plot
        if 'area' in df.columns and 'cell_count' in df.columns:
            plt.figure(figsize=(10, 6))
            plt.scatter(df['cell_count'], df['area'])
            plt.title('Area vs Cell Count')
            plt.xlabel('Cell Count')
            plt.ylabel('Area')
            plt.grid(True)
            plt.savefig(os.path.join(plots_dir, 'area_vs_cell_count.png'))
            print(f"Created plot: area_vs_cell_count.png")
        
        # Runtime histogram
        if 'runtime_total_seconds' in df.columns:
            plt.figure(figsize=(10, 6))
            plt.hist(df['runtime_total_seconds'], bins=20)
            plt.title('Runtime Distribution')
            plt.xlabel('Runtime (seconds)')
            plt.ylabel('Frequency')
            plt.grid(True)
            plt.savefig(os.path.join(plots_dir, 'runtime_histogram.png'))
            print(f"Created plot: runtime_histogram.png")
        
        # Power comparison bar chart (top 10 by total power)
        if 'total_power' in df.columns:
            top_power = df.sort_values('total_power', ascending=False).head(10)
            plt.figure(figsize=(12, 6))
            plt.bar(top_power['design'], top_power['total_power'])
            plt.title('Top 10 Designs by Power Consumption')
            plt.xlabel('Design')
            plt.ylabel('Total Power')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'top_power_designs.png'))
            print(f"Created plot: top_power_designs.png")
        
    except Exception as e:
        print(f"Error creating plots: {str(e)}")
    
    print(f"\nAnalysis complete. DataFrame contains {len(df)} designs with {len(df.columns)} metrics.")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 