import pyrtl
from pyrtl import WireVector, Input, Output
import pyrtl.analysis as analysis
import os
import sys
import urllib.request
import tempfile
import numpy as np

# Add the parent directory to the path so we can import from hardware_accelerators
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from hardware_accelerators.rtllib.adders import float_adder
from hardware_accelerators.rtllib.multipliers import float_multiplier
from hardware_accelerators.dtypes import Float16, Float32, Float8, BF16
from hardware_accelerators.analysis.pipeline_multiply_add import create_pipeline_multiply_add


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


def create_simple_adder(data_type):
    """
    Create a simple adder circuit.
    
    Args:
        data_type: The floating-point data type to use
        
    Returns:
        None
    """
    # Clear any existing PyRTL design
    pyrtl.reset_working_block()
    
    # Create input and output wires
    a = Input(data_type.bitwidth(), 'a')
    b = Input(data_type.bitwidth(), 'b')
    result = Output(data_type.bitwidth(), 'result')
    
    # Create adder
    result <<= float_adder(a, b, data_type)
    
    return pyrtl.working_block()


def create_simple_multiplier(data_type):
    """
    Create a simple multiplier circuit.
    
    Args:
        data_type: The floating-point data type to use
        
    Returns:
        None
    """
    # Clear any existing PyRTL design
    pyrtl.reset_working_block()
    
    # Create input and output wires
    a = Input(data_type.bitwidth(), 'a')
    b = Input(data_type.bitwidth(), 'b')
    result = Output(data_type.bitwidth(), 'result')
    
    # Create multiplier
    result <<= float_multiplier(a, b, data_type)
    
    return pyrtl.working_block()


def create_multiply_add(data_type):
    """
    Create a simple multiply-add circuit (a*b + c).
    
    Args:
        data_type: The floating-point data type to use
        
    Returns:
        None
    """
    # Clear any existing PyRTL design
    pyrtl.reset_working_block()
    
    # Create input and output wires
    a = Input(data_type.bitwidth(), 'a')
    b = Input(data_type.bitwidth(), 'b')
    c = Input(data_type.bitwidth(), 'c')
    result = Output(data_type.bitwidth(), 'result')
    
    # Create multiply-add circuit
    mult_result = float_multiplier(a, b, data_type)
    result <<= float_adder(mult_result, c, data_type)
    
    return pyrtl.working_block()


def analyze_with_yosys(block, liberty_file, data_type_name):
    """
    Analyze the given block with Yosys using the provided Liberty file.
    
    Args:
        block: PyRTL block to analyze
        liberty_file: Path to the Liberty file
        data_type_name: Name of the data type used
        
    Returns:
        area: Area of the circuit
        delay: Delay of the circuit
    """
    print(f"\nAnalyzing {data_type_name} circuit with Yosys...")
    
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
    """Main function to analyze circuits with Yosys."""
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
        "adder": {},
        "multiplier": {},
        "multiply_add": {},
        "pipeline_accelerator": {}
    }
    
    # Analyze adders
    print("\n=== Analyzing Adders ===")
    for dtype, dtype_name in data_types:
        block = create_simple_adder(dtype)
        area, delay = analyze_with_yosys(block, liberty_file, f"{dtype_name} Adder")
        results["adder"][dtype_name] = {"area": area, "delay": delay}
    
    # Analyze multipliers
    print("\n=== Analyzing Multipliers ===")
    for dtype, dtype_name in data_types:
        block = create_simple_multiplier(dtype)
        area, delay = analyze_with_yosys(block, liberty_file, f"{dtype_name} Multiplier")
        results["multiplier"][dtype_name] = {"area": area, "delay": delay}
    
    # Analyze multiply-add circuits
    print("\n=== Analyzing Multiply-Add Circuits ===")
    for dtype, dtype_name in data_types:
        block = create_multiply_add(dtype)
        area, delay = analyze_with_yosys(block, liberty_file, f"{dtype_name} Multiply-Add")
        results["multiply_add"][dtype_name] = {"area": area, "delay": delay}
    
    # Analyze pipeline accelerators
    print("\n=== Analyzing Pipeline Accelerators ===")
    for dtype, dtype_name in data_types:
        # Create pipeline accelerator
        accelerator = create_pipeline_multiply_add(dtype, array_size=2, pipeline=True)
        # Get the working block
        block = pyrtl.working_block()
        area, delay = analyze_with_yosys(block, liberty_file, f"{dtype_name} Pipeline Accelerator")
        results["pipeline_accelerator"][dtype_name] = {"area": area, "delay": delay}
    
    # Print summary
    print("\n=== Summary ===")
    for circuit_type, circuit_results in results.items():
        print(f"\n{circuit_type.upper()}:")
        for dtype_name, metrics in circuit_results.items():
            print(f"  {dtype_name}: Area = {metrics['area']}, Delay = {metrics['delay']} ns")
    
    return results


if __name__ == "__main__":
    main() 