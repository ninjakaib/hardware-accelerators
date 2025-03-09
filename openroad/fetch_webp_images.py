#!/usr/bin/env python3
import os
import shutil
import subprocess
import sys
from data_transfer import SOURCE_DIR, SOURCE_DIR2, WEBP_DIR, WSL_REPORTS_PATH, ensure_directory

def main():
    # Check if source directories exist
    if not os.path.exists(SOURCE_DIR) and not os.path.exists(SOURCE_DIR2):
        print(f"Error: Neither source directory '{SOURCE_DIR}' nor '{SOURCE_DIR2}' exists.")
        print("Please run the organize_verilog.py script first to create the source directories.")
        return 1
    
    # Check if WSL WebP path exists and is accessible
    if not os.path.exists(WSL_REPORTS_PATH):
        print(f"Error: WSL WebP path '{WSL_REPORTS_PATH}' does not exist or is not accessible.")
        print("Please make sure the WSL path is correct and accessible from Windows.")
        return 1
    
    # Create WebP images directory if it doesn't exist
    ensure_directory(WEBP_DIR)
    
    # Get list of designs from both source directories
    designs = []
    if os.path.exists(SOURCE_DIR):
        designs.extend([d for d in os.listdir(SOURCE_DIR) if os.path.isdir(os.path.join(SOURCE_DIR, d))])
    if os.path.exists(SOURCE_DIR2):
        designs.extend([d for d in os.listdir(SOURCE_DIR2) if os.path.isdir(os.path.join(SOURCE_DIR2, d))])
    
    # Remove duplicates
    designs = list(set(designs))
    
    if not designs:
        print("No designs found in the source directories.")
        return 1
    
    # Copy and rename WebP images for each design
    copied_count = 0
    for design in designs:
        # Source path for the WebP image in WSL
        webp_source_path = os.path.join(WSL_REPORTS_PATH, design, "base", "final_resized.webp")
        
        # Destination path with renamed file
        webp_dest_path = os.path.join(WEBP_DIR, f"{design}_final.webp")
        
        # Check if source WebP image exists
        if os.path.exists(webp_source_path):
            print(f"Copying WebP image for {design}...")
            try:
                shutil.copy2(webp_source_path, webp_dest_path)
                copied_count += 1
                print(f"  Saved to: {webp_dest_path}")
            except Exception as e:
                print(f"  Error copying WebP image for {design}: {str(e)}")
        else:
            print(f"WebP image not found for {design}: {webp_source_path}")
    
    print(f"\nCopied {copied_count} out of {len(designs)} WebP images to {WEBP_DIR}")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 