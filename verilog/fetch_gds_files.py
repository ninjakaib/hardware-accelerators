#!/usr/bin/env python3
import os
import shutil
import subprocess
import sys

def main():
    # Script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Source directories (where the "all" and "art" folders are located)
    source_dir = os.path.join(script_dir, "all")
    source_dir2 = os.path.join(script_dir, "art")
    
    # GDS files directory (where we'll save the fetched GDS files)
    gds_dir = os.path.join(script_dir, "gds_files")
    
    # WSL path to the OpenROAD GDS files
    wsl_gds_path = r"\\wsl$\Ubuntu\home\user\OpenROAD-flow-scripts\flow\results\nangate45"
    
    # Check if source directories exist
    if not os.path.exists(source_dir):
        print(f"Error: Source directory '{source_dir}' does not exist.")
        print("Please run the organize_verilog.py script first to create the 'all' directory.")
        return 1
    
    # Check if WSL GDS path exists and is accessible
    if not os.path.exists(wsl_gds_path):
        print(f"Error: WSL GDS path '{wsl_gds_path}' does not exist or is not accessible.")
        print("Please make sure the WSL path is correct and accessible from Windows.")
        return 1
    
    # Create GDS files directory if it doesn't exist
    if not os.path.exists(gds_dir):
        os.makedirs(gds_dir)
        print(f"Created GDS files directory: {gds_dir}")
    
    # Get list of subdirectories in the "all" and "art" folders (these are our designs)
    designs = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
    designs2 = []
    if os.path.exists(source_dir2):
        designs2 = [d for d in os.listdir(source_dir2) if os.path.isdir(os.path.join(source_dir2, d))]

    designs = designs + designs2
    
    if not designs:
        print("No designs found in the 'all' or 'art' directories.")
        return 1
    
    # Copy and rename 6_final.gds for each design
    copied_count = 0
    for design in designs:
        # Source path for the 6_final.gds in WSL
        gds_source_path = os.path.join(wsl_gds_path, design, "base", "6_final.gds")
        
        # Destination path with renamed file (design name + 6_final.gds)
        gds_dest_path = os.path.join(gds_dir, f"{design}_6_final.gds")
        
        # Check if source GDS file exists
        if os.path.exists(gds_source_path):
            print(f"Copying GDS file for {design}...")
            try:
                shutil.copy2(gds_source_path, gds_dest_path)
                copied_count += 1
                print(f"  Saved to: {gds_dest_path}")
            except Exception as e:
                print(f"  Error copying GDS file for {design}: {str(e)}")
        else:
            print(f"GDS file not found for {design}: {gds_source_path}")
    
    print(f"\nCopied {copied_count} out of {len(designs)} GDS files to {gds_dir}")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 