#!/usr/bin/env python3
"""
Run all analysis scripts in sequence.
"""

import os
import sys
import importlib.util
import time

def import_module_from_path(module_name, file_path):
    """Import a module from a file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    """Run all analysis scripts in sequence."""
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Add the parent directory to the path
    sys.path.append(os.path.dirname(os.path.dirname(script_dir)))
    
    print("=" * 80)
    print("Running Pipeline Multiply-Add Analysis")
    print("=" * 80)
    
    # Import and run pipeline_multiply_add.py
    pipeline_module_path = os.path.join(script_dir, "pipeline_multiply_add.py")
    pipeline_module = import_module_from_path("pipeline_multiply_add", pipeline_module_path)
    accelerators = pipeline_module.main()
    
    # Wait a moment to separate outputs
    time.sleep(1)
    
    print("\n" + "=" * 80)
    print("Running Yosys Area-Delay Analysis")
    print("=" * 80)
    
    # Import and run yosys_analysis.py
    yosys_module_path = os.path.join(script_dir, "yosys_analysis.py")
    yosys_module = import_module_from_path("yosys_analysis", yosys_module_path)
    results = yosys_module.main()
    
    print("\n" + "=" * 80)
    print("Analysis Complete")
    print("=" * 80)
    
    return accelerators, results


if __name__ == "__main__":
    main() 