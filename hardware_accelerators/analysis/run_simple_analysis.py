#!/usr/bin/env python3
"""
Run analysis on simplified circuits (a+b and a*b).
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
    """Run analysis on simplified circuits."""
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Add the parent directory to the path
    sys.path.append(os.path.dirname(os.path.dirname(script_dir)))
    
    print("=" * 80)
    print("Running Simple Circuits Analysis")
    print("=" * 80)
    
    # Import and run simple_circuits.py
    simple_circuits_path = os.path.join(script_dir, "simple_circuits.py")
    simple_circuits_module = import_module_from_path("simple_circuits", simple_circuits_path)
    circuit_results = simple_circuits_module.main()
    
    # Wait a moment to separate outputs
    time.sleep(1)
    
    print("\n" + "=" * 80)
    print("Running Yosys Area-Delay Analysis on Simple Circuits")
    print("=" * 80)
    
    # Import and run simple_yosys_analysis.py
    yosys_module_path = os.path.join(script_dir, "simple_yosys_analysis.py")
    yosys_module = import_module_from_path("simple_yosys_analysis", yosys_module_path)
    yosys_results = yosys_module.main()
    
    print("\n" + "=" * 80)
    print("Simple Circuits Analysis Complete")
    print("=" * 80)
    
    return circuit_results, yosys_results


if __name__ == "__main__":
    main() 