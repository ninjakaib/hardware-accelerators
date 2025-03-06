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
    source_dir2 = os.path.join(script_dir, "art")
    
    # WebP images directory (where we'll save the fetched WebP images)
    webp_dir = os.path.join(script_dir, "webp_images")
    
    # WSL path to the OpenROAD WebP images
    wsl_webp_path = r"\\wsl$\Ubuntu\home\user\OpenROAD-flow-scripts\flow\reports\nangate45"
    
    # Check if source directory exists
    if not os.path.exists(source_dir):
        print(f"Error: Source directory '{source_dir}' does not exist.")
        print("Please run the organize_verilog.py script first to create the 'all' directory.")
        return 1
    
    # Check if WSL WebP path exists and is accessible
    if not os.path.exists(wsl_webp_path):
        print(f"Error: WSL WebP path '{wsl_webp_path}' does not exist or is not accessible.")
        print("Please make sure the WSL path is correct and accessible from Windows.")
        return 1
    
    # Create WebP images directory if it doesn't exist
    if not os.path.exists(webp_dir):
        os.makedirs(webp_dir)
        print(f"Created WebP images directory: {webp_dir}")
    
    # Get list of subdirectories in the "all" folder (these are our designs)
    designs = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
    designs2 = [d for d in os.listdir(source_dir2) if os.path.isdir(os.path.join(source_dir2, d))]

    designs = designs + designs2
    
    if not designs:
        print("No designs found in the 'all' directory.")
        return 1
    
    # Copy and rename final_all.webp for each design
    copied_count = 0
    for design in designs:
        # Source path for the final_all.webp in WSL
        webp_source_path = os.path.join(wsl_webp_path, design, "base", "final_all.webp")
        
        # Destination path with renamed file (just the design name)
        webp_dest_path = os.path.join(webp_dir, f"{design}.webp")
        
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
    
    print(f"\nCopied {copied_count} out of {len(designs)} WebP images to {webp_dir}")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 