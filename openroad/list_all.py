#!/usr/bin/env python3
"""
Script to list all directories in the 'all' subdirectory.
"""

import os
import sys
from pathlib import Path


def list_directories(directory_path):
    """
    List all directories within the given directory path.

    Args:
        directory_path (str): Path to the directory to scan

    Returns:
        list: List of directory names
    """
    try:
        # Get the absolute path
        abs_path = Path(directory_path).resolve()

        # Check if the directory exists
        if not abs_path.exists():
            print(f"Error: Directory '{directory_path}' does not exist.")
            return []

        # Check if it's a directory
        if not abs_path.is_dir():
            print(f"Error: '{directory_path}' is not a directory.")
            return []

        # Get all directories (exclude files)
        directories = [d.name for d in abs_path.iterdir() if d.is_dir()]

        return sorted(directories)

    except Exception as e:
        print(f"Error: {e}")
        return []


def main():
    # Path to the 'all' subdirectory
    all_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "verilog", "all"
    )

    # Get list of directories
    directories = list_directories(all_dir)

    # Print the results
    if directories:
        print(f"Directories in '{all_dir}':")
        for i, directory in enumerate(directories, 1):
            print(f"{i}. {directory}")
        print(f"\nTotal: {len(directories)} directories")
    else:
        print(f"No directories found in '{all_dir}'")


if __name__ == "__main__":
    main()
