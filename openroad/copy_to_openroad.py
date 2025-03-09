#!/usr/bin/env python3
import os
import shutil
import subprocess
import sys
from data_transfer import SOURCE_DIR, WSL_DESIGNS_PATH

def main():
    # Check if source directory exists
    if not os.path.exists(SOURCE_DIR):
        print(f"Error: Source directory '{SOURCE_DIR}' does not exist.")
        print("Please run the organize_verilog.py script first to create the 'all' directory.")
        return 1
    
    # Check if destination directory exists and is accessible
    if not os.path.exists(WSL_DESIGNS_PATH):
        print(f"Error: Destination directory '{WSL_DESIGNS_PATH}' does not exist or is not accessible.")
        print("Please make sure the WSL path is correct and accessible from Windows.")
        return 1
    
    # Get list of subdirectories in the "all" folder
    subdirs = [d for d in os.listdir(SOURCE_DIR) if os.path.isdir(os.path.join(SOURCE_DIR, d))]
    
    if not subdirs:
        print("No subdirectories found in the 'all' directory.")
        return 1
    
    # Copy each subdirectory to the destination
    copied_count = 0
    for subdir in subdirs:
        source_subdir = os.path.join(SOURCE_DIR, subdir)
        dest_subdir = os.path.join(WSL_DESIGNS_PATH, subdir)
        
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
    
    print(f"\nCopied {copied_count} out of {len(subdirs)} directories to {WSL_DESIGNS_PATH}")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 