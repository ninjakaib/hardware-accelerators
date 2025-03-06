#!/usr/bin/env python3
import os
import shutil
import subprocess
import sys

def main():
    # Script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Source directory (where the "all" folder is located)
    source_dir = os.path.join(script_dir, "all")
    
    # Reports directory (where we'll save the fetched reports)
    reports_dir = os.path.join(script_dir, "reports")
    
    # WSL path to the OpenROAD logs
    wsl_logs_path = r"\\wsl$\Ubuntu\home\user\OpenROAD-flow-scripts\flow\logs\nangate45"
    
    # Check if source directory exists
    if not os.path.exists(source_dir):
        print(f"Error: Source directory '{source_dir}' does not exist.")
        print("Please run the organize_verilog.py script first to create the 'all' directory.")
        return 1
    
    # Check if WSL logs path exists and is accessible
    if not os.path.exists(wsl_logs_path):
        print(f"Error: WSL logs path '{wsl_logs_path}' does not exist or is not accessible.")
        print("Please make sure the WSL path is correct and accessible from Windows.")
        return 1
    
    # Create reports directory if it doesn't exist
    if not os.path.exists(reports_dir):
        os.makedirs(reports_dir)
        print(f"Created reports directory: {reports_dir}")
    
    # Get list of subdirectories in the "all" folder (these are our designs)
    designs = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
    
    if not designs:
        print("No designs found in the 'all' directory.")
        return 1
    
    # Copy and rename report.json for each design
    copied_count = 0
    for design in designs:
        # Source path for the report.json in WSL
        report_source_path = os.path.join(wsl_logs_path, design, "base", "6_report.json")
        
        # Destination path with renamed file
        report_dest_path = os.path.join(reports_dir, f"{design}_report.json")
        
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
    
    print(f"\nCopied {copied_count} out of {len(designs)} reports to {reports_dir}")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 