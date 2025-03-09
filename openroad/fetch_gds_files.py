#!/usr/bin/env python3
import os
import shutil
import subprocess
import sys
from data_transfer import (
    SOURCE_DIR,
    SOURCE_DIR2,
    GDS_DIR,
    WSL_RESULTS_PATH,
    ensure_directory,
)


def main():
    # Check if source directories exist
    if not os.path.exists(SOURCE_DIR) and not os.path.exists(SOURCE_DIR2):
        print(
            f"Error: Neither source directory '{SOURCE_DIR}' nor '{SOURCE_DIR2}' exists."
        )
        print(
            "Please run the organize_verilog.py script first to create the source directories."
        )
        return 1

    # Check if WSL GDS path exists and is accessible
    if not os.path.exists(WSL_RESULTS_PATH):
        print(
            f"Error: WSL GDS path '{WSL_RESULTS_PATH}' does not exist or is not accessible."
        )
        print("Please make sure the WSL path is correct and accessible from Windows.")
        return 1

    # Create GDS files directory if it doesn't exist
    ensure_directory(GDS_DIR)

    # Get list of designs from both source directories
    designs = []
    if os.path.exists(SOURCE_DIR):
        designs.extend(
            [
                d
                for d in os.listdir(SOURCE_DIR)
                if os.path.isdir(os.path.join(SOURCE_DIR, d))
            ]
        )
    if os.path.exists(SOURCE_DIR2):
        designs.extend(
            [
                d
                for d in os.listdir(SOURCE_DIR2)
                if os.path.isdir(os.path.join(SOURCE_DIR2, d))
            ]
        )

    # Remove duplicates
    designs = list(set(designs))

    if not designs:
        print("No designs found in the source directories.")
        return 1

    # Copy and rename GDS files for each design
    copied_count = 0
    for design in designs:
        # Source path for the GDS file in WSL
        gds_source_path = os.path.join(WSL_RESULTS_PATH, design, "base", "6_final.gds")

        # Destination path with renamed file
        gds_dest_path = os.path.join(GDS_DIR, f"{design}_final.gds")

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

    print(f"\nCopied {copied_count} out of {len(designs)} GDS files to {GDS_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
