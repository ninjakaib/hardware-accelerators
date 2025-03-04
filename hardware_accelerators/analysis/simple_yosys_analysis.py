import pyrtl
import os
import sys
import urllib.request
import numpy as np

# Add the parent directory to the path so we can import from hardware_accelerators
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from hardware_accelerators.dtypes import Float16, Float32, Float8, BF16
from hardware_accelerators.analysis.simple_circuits import (
    create_simple_adder,
    create_simple_multiplier,
    create_pipelined_adder,
    create_pipelined_multiplier
)

try:
    import pyrtl.analysis as analysis
except ImportError:
    print("Warning: pyrtl.analysis module not found. Yosys analysis will not be available.")
    analysis = None


def download_liberty_file(url, output_path):
    """
    Download the Liberty file from the given URL.
    
    Args:
        url: URL of the Liberty file
        output_path: Path to save the Liberty file
        
    Returns:
        output_path: Path to the downloaded Liberty file
    """
    print(f"Downloading Liberty file from {url}...")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Download the file
    urllib.request.urlretrieve(url, output_path)
    
    print(f"Liberty file downloaded to {output_path}")
    return output_path


def analyze_with_yosys(block, liberty_file, circuit_name):
    """
    Analyze the given block with Yosys using the provided Liberty file.
    
    Args:
        block: PyRTL block to analyze
        liberty_file: Path to the Liberty file
        circuit_name: Name of the circuit being analyzed
        
    Returns:
        area: Area of the circuit
        delay: Delay of the circuit
    """
    print(f"\nAnalyzing {circuit_name} with Yosys...")
    
    if analysis is None:
        print("Skipping Yosys analysis as pyrtl.analysis module is not available.")
        return None, None
    
    try:
        # Use the yosys_area_delay function from PyRTL analysis
        area, delay = analysis.yosys_area_delay(library=liberty_file, block=block)
        print(f"Area: {area}")
        print(f"Delay: {delay} ns")
        return area, delay
    except Exception as e:
        print(f"Error analyzing circuit with Yosys: {e}")
        return None, None


def main():
    """Main function to analyze simplified circuits with Yosys."""
    # URL of the Liberty file
    liberty_url = "https://raw.githubusercontent.com/mflowgen/freepdk-45nm/master/stdcells-bc.lib"
    
    # Path to save the Liberty file
    liberty_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "freepdk-45nm-stdcells-bc.lib")
    
    # Download the Liberty file if it doesn't exist
    if not os.path.exists(liberty_file):
        liberty_file = download_liberty_file(liberty_url, liberty_file)
    
    # Data types to analyze
    data_types = [
        (Float8, "Float8"),
        (BF16, "BF16"),
        (Float16, "Float16"),
        (Float32, "Float32")
    ]
    
    # Results dictionary
    results = {
        "simple_adder": {},
        "simple_multiplier": {},
        "pipelined_adder": {},
        "pipelined_multiplier": {}
    }
    
    # Analyze simple adders
    print("\n=== Analyzing Simple Adders ===")
    for dtype, dtype_name in data_types:
        block = create_simple_adder(dtype)
        area, delay = analyze_with_yosys(block, liberty_file, f"{dtype_name} Simple Adder")
        results["simple_adder"][dtype_name] = {"area": area, "delay": delay}
    
    # Analyze simple multipliers
    print("\n=== Analyzing Simple Multipliers ===")
    for dtype, dtype_name in data_types:
        block = create_simple_multiplier(dtype)
        area, delay = analyze_with_yosys(block, liberty_file, f"{dtype_name} Simple Multiplier")
        results["simple_multiplier"][dtype_name] = {"area": area, "delay": delay}
    
    # Analyze pipelined adders
    print("\n=== Analyzing Pipelined Adders ===")
    for dtype, dtype_name in data_types:
        block, _ = create_pipelined_adder(dtype)
        area, delay = analyze_with_yosys(block, liberty_file, f"{dtype_name} Pipelined Adder")
        results["pipelined_adder"][dtype_name] = {"area": area, "delay": delay}
    
    # Analyze pipelined multipliers
    print("\n=== Analyzing Pipelined Multipliers ===")
    for dtype, dtype_name in data_types:
        block, _ = create_pipelined_multiplier(dtype)
        area, delay = analyze_with_yosys(block, liberty_file, f"{dtype_name} Pipelined Multiplier")
        results["pipelined_multiplier"][dtype_name] = {"area": area, "delay": delay}
    
    # Print summary
    print("\n=== Summary ===")
    for circuit_type, circuit_results in results.items():
        print(f"\n{circuit_type.upper()}:")
        for dtype_name, metrics in circuit_results.items():
            print(f"  {dtype_name}: Area = {metrics['area']}, Delay = {metrics['delay']} ns")
    
    return results


if __name__ == "__main__":
    main() 