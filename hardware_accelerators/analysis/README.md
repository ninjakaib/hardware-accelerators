# Hardware Accelerator Analysis

This directory contains scripts for analyzing hardware accelerators using PyRTL and Yosys.

## Overview

The scripts in this directory allow you to:

1. Create and simulate a simple pipeline multiply-add accelerator using the CompiledAccelerator class from rtllib.
2. Analyze the area and delay of various floating-point circuits using the Yosys synthesis tool with the FreePDK-45nm standard cell library.

## Prerequisites

- PyRTL (Python Register Transfer Level)
- Yosys synthesis tool (for area and delay analysis)
- Python 3.6+
- NumPy

## Scripts

### 1. `pipeline_multiply_add.py`

This script creates a simple pipeline multiply-add accelerator using the CompiledAccelerator class from rtllib. It uses the adders and multipliers implemented in rtllib.

**Features:**

- Creates a pipeline multiply-add accelerator with configurable data types
- Supports different floating-point formats (Float8, BF16, Float16, Float32)
- Simulates the accelerator with random input data

**Usage:**

```python
python pipeline_multiply_add.py
```

### 2. `yosys_analysis.py`

This script analyzes the area and delay of various floating-point circuits using the Yosys synthesis tool with the FreePDK-45nm standard cell library.

**Features:**

- Downloads the FreePDK-45nm standard cell library from GitHub
- Creates and analyzes simple adder, multiplier, and multiply-add circuits
- Analyzes the pipeline accelerator created by `pipeline_multiply_add.py`
- Supports different floating-point formats (Float8, BF16, Float16, Float32)
- Provides a summary of area and delay metrics for each circuit

**Usage:**

```python
python yosys_analysis.py
```

### 3. `run_analysis.py`

This script runs both the pipeline multiply-add analysis and the Yosys area-delay analysis in sequence.

**Features:**

- Runs all analysis scripts in a single command
- Collects and returns results from both analyses
- Provides clear separation between different analysis outputs

**Usage:**

```python
python run_analysis.py
```

## How It Works

### Pipeline Multiply-Add Accelerator

The `pipeline_multiply_add.py` script creates a simple pipeline multiply-add accelerator using the CompiledAccelerator class from rtllib. The accelerator is configured with the following components:

- A systolic array of configurable size (default: 2x2)
- Floating-point adders and multipliers from rtllib
- Accumulator and activation units

The script simulates the accelerator with random input data and prints the results.

### Yosys Area-Delay Analysis

The `yosys_analysis.py` script uses the PyRTL `yosys_area_delay` function to analyze the area and delay of various floating-point circuits. It creates the following circuits:

1. Simple adder circuits using `float_adder` from rtllib
2. Simple multiplier circuits using `float_multiplier` from rtllib
3. Simple multiply-add circuits (a\*b + c) using both `float_adder` and `float_multiplier`
4. Pipeline accelerators created by `pipeline_multiply_add.py`

The script analyzes each circuit with the FreePDK-45nm standard cell library and prints a summary of the area and delay metrics.

## Notes

- The FreePDK-45nm standard cell library is downloaded from GitHub if it doesn't exist locally.
- The Yosys synthesis tool must be installed and accessible in the system PATH for the area-delay analysis to work.
- The scripts use the PyRTL `yosys_area_delay` function, which requires Yosys to be installed.
- If you encounter issues with the Yosys analysis, make sure that:
  1. Yosys is installed and in your PATH
  2. The FreePDK-45nm library is compatible with your version of Yosys
  3. PyRTL's yosys_area_delay function is properly configured

## References

- [PyRTL Documentation](https://pyrtl.readthedocs.io/en/latest/)
- [PyRTL Analysis Documentation](https://pyrtl.readthedocs.io/en/latest/analysis.html)
- [FreePDK-45nm Standard Cell Library](https://github.com/mflowgen/freepdk-45nm/blob/master/stdcells-bc.lib)
- [Yosys Synthesis Tool](http://www.clifford.at/yosys/)
