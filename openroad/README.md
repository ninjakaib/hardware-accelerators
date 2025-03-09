# OpenROAD Integration Tools

This directory contains scripts and utilities for integrating with the OpenROAD electronic design automation (EDA) tool suite to analyze and optimize hardware accelerator designs.

## Overview

The OpenROAD tools in this directory facilitate the synthesis, placement, routing, and analysis of hardware designs created in the project. These scripts help bridge the gap between Verilog designs and physical implementation metrics such as power, area, and timing.

## Key Components

- **analyze_hardware.py**: Comprehensive analysis script that fetches and processes JSON reports and timing data from OpenROAD runs
- **build_designs.sh**: Shell script to build all hardware designs in the designs directory
- **copy_to_openroad.py**: Utility to copy Verilog files to the OpenROAD environment
- **fetch_gds_files.py**: Script to retrieve GDS layout files from OpenROAD runs
- **fetch_webp_images.py**: Script to retrieve layout images in WebP format
- **list_all.py**: Utility to list all available hardware designs
- **organize_verilog.py**: Script to organize and prepare Verilog files for OpenROAD processing
- **run_all.py**: Orchestration script to run the complete OpenROAD workflow

### Data Transfer Module

The `data_transfer` subdirectory contains configuration and utilities for transferring data between the main project environment and the OpenROAD environment:

- **config.py**: Configuration file defining paths and directories for data transfer
- ****init**.py**: Module initialization file with utility functions

## Prerequisites

- OpenROAD flow scripts installed in WSL (Windows Subsystem for Linux)
- Python 3.6+ with pandas and other dependencies listed in the project's requirements.txt
- Proper configuration of WSL paths in `data_transfer/config.py`

## Usage

### Analyzing Hardware Designs

```bash
python analyze_hardware.py
```

This will fetch reports from WSL, extract timing information, process area and power data, and generate a comprehensive analysis CSV file.

### Building All Designs

```bash
./build_designs.sh
```

This script will iterate through all design subdirectories and build them using their respective configurations.

### Organizing Verilog Files

```bash
python organize_verilog.py
```

This will organize Verilog files from the source directory into the appropriate structure for OpenROAD processing.

### Running the Complete Workflow

```bash
python run_all.py
```

This script orchestrates the entire process from organizing Verilog files to analyzing the results.

## Integration with Main Project

The OpenROAD tools integrate with the hardware accelerator designs in the `verilog` directory. The analysis results provide valuable insights into the physical implementation characteristics of the hardware accelerators, including:

- Power consumption
- Area utilization
- Timing performance
- Layout visualization

These metrics are essential for evaluating the efficiency and feasibility of the L-Mul algorithm and other hardware optimizations developed in this project.

## Notes

- Ensure that the WSL paths in `data_transfer/config.py` are correctly configured for your environment
- The scripts assume a specific directory structure in the OpenROAD-flow-scripts installation
- Some scripts may require administrative privileges to access WSL file systems
