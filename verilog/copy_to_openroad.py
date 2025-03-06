#!/usr/bin/env python3
import os
import shutil
import subprocess
import sys

def main():
    # Source directory (where the "all" folder is located)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    source_dir = os.path.join(script_dir, "all")
    
    # Destination directory in WSL
    wsl_destination = r"\\wsl$\Ubuntu\home\user\OpenROAD-flow-scripts\flow\designs"
    
    # Check if source directory exists
    if not os.path.exists(source_dir):
        print(f"Error: Source directory '{source_dir}' does not exist.")
        print("Please run the organize_verilog.py script first to create the 'all' directory.")
        return 1
    
    # Check if destination directory exists and is accessible
    if not os.path.exists(wsl_destination):
        print(f"Error: Destination directory '{wsl_destination}' does not exist or is not accessible.")
        print("Please make sure the WSL path is correct and accessible from Windows.")
        return 1
    
    # Get list of subdirectories in the "all" folder
    subdirs = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
    
    if not subdirs:
        print("No subdirectories found in the 'all' directory.")
        return 1
    
    # Copy each subdirectory to the destination
    copied_count = 0
    for subdir in subdirs:
        source_subdir = os.path.join(source_dir, subdir)
        dest_subdir = os.path.join(wsl_destination, subdir)
        
        # Remove destination directory if it already exists
        if os.path.exists(dest_subdir):
            print(f"Removing existing directory: {dest_subdir}")
            shutil.rmtree(dest_subdir)
        
        # Copy the directory
        print(f"Copying {source_subdir} to {dest_subdir}")
        try:
            shutil.copytree(source_subdir, dest_subdir)
            copied_count += 1
        except Exception as e:
            print(f"Error copying {subdir}: {str(e)}")
    
    print(f"\nCopied {copied_count} out of {len(subdirs)} directories to {wsl_destination}")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 