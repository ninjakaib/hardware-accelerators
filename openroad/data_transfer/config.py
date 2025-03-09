#!/usr/bin/env python3
"""
Configuration file for data transfer paths.
Contains absolute paths for source and destination directories used in various scripts.
"""

import os

# Base workspace directory - set relative to project root
# Get the directory of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))
# Navigate up two levels (from openroad/data_transfer to project root)
WORKSPACE_DIR = os.path.abspath(os.path.join(current_dir, "..", ".."))

# Verilog directory
VERILOG_DIR = os.path.join(WORKSPACE_DIR, "verilog")

# Source directories
SOURCE_DIR = os.path.join(VERILOG_DIR, "all")
SOURCE_DIR2 = os.path.join(VERILOG_DIR, "art")

# Output directories
REPORTS_DIR = os.path.join(current_dir, "reports")
FINISH_REPORTS_DIR = os.path.join(current_dir, "finish_reports")
GDS_DIR = os.path.join(current_dir, "gds_files")
WEBP_DIR = os.path.join(current_dir, "webp_images")

# WSL paths
#set this to the path of the OpenROAD-flow-scripts directory
WSL_OPENROAD_BASE = r"\\wsl$\Ubuntu\home\user\OpenROAD-flow-scripts"
WSL_DESIGNS_PATH = os.path.join(WSL_OPENROAD_BASE, "flow", "designs")
WSL_LOGS_PATH = os.path.join(WSL_OPENROAD_BASE, "flow", "logs", "nangate45")
WSL_REPORTS_PATH = os.path.join(WSL_OPENROAD_BASE, "flow", "reports", "nangate45")
WSL_RESULTS_PATH = os.path.join(WSL_OPENROAD_BASE, "flow", "results", "nangate45")

# Function to ensure directories exist
def ensure_directory(directory_path):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Created directory: {directory_path}")
    return directory_path 