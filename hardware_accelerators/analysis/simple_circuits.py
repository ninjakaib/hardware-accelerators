import pyrtl
from pyrtl import WireVector, Input, Output, Simulation
import numpy as np
import sys
import os

# Add the parent directory to the path so we can import from hardware_accelerators
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from hardware_accelerators.dtypes import Float16, Float32, Float8, BF16


def create_simple_adder(data_type):
    """
    Create a simple adder circuit (a + b) using PyRTL's built-in operators.
    
    Args:
        data_type: The data type to use (only used for bitwidth)
        
    Returns:
        The PyRTL working block
    """
    # Clear any existing PyRTL design
    pyrtl.reset_working_block()
    
    # Create input and output wires
    a = Input(data_type.bitwidth(), 'a')
    b = Input(data_type.bitwidth(), 'b')
    result = Output(data_type.bitwidth(), 'result')
    
    # Create adder using PyRTL's built-in addition
    # Note: This treats the inputs as unsigned integers, not floating point
    result <<= a + b
    
    return pyrtl.working_block()


def create_simple_multiplier(data_type):
    """
    Create a simple multiplier circuit (a * b) using PyRTL's built-in operators.
    
    Args:
        data_type: The data type to use (only used for bitwidth)
        
    Returns:
        The PyRTL working block
    """
    # Clear any existing PyRTL design
    pyrtl.reset_working_block()
    
    # Create input and output wires
    a = Input(data_type.bitwidth(), 'a')
    b = Input(data_type.bitwidth(), 'b')
    result = Output(data_type.bitwidth(), 'result')
    
    # Create multiplier using PyRTL's built-in multiplication
    # Note: This treats the inputs as unsigned integers, not floating point
    # We'll truncate the result to match the input bitwidth
    mult_result = a * b
    result <<= mult_result[:data_type.bitwidth()]
    
    return pyrtl.working_block()


def create_pipelined_adder(data_type):
    """
    Create a pipelined adder circuit (a + b) using PyRTL's built-in operators.
    
    Args:
        data_type: The data type to use (only used for bitwidth)
        
    Returns:
        The PyRTL working block and the result wire
    """
    # Clear any existing PyRTL design
    pyrtl.reset_working_block()
    
    # Create input and output wires
    a = Input(data_type.bitwidth(), 'a')
    b = Input(data_type.bitwidth(), 'b')
    result = Output(data_type.bitwidth(), 'result')
    
    # Create pipeline registers
    a_reg = pyrtl.Register(bitwidth=data_type.bitwidth(), name='a_reg')
    b_reg = pyrtl.Register(bitwidth=data_type.bitwidth(), name='b_reg')
    
    # Connect input to registers
    a_reg.next <<= a
    b_reg.next <<= b
    
    # Perform addition in the next stage using PyRTL's built-in addition
    add_result = a_reg + b_reg
    
    # Connect to output
    result <<= add_result
    
    return pyrtl.working_block(), result


def create_pipelined_multiplier(data_type):
    """
    Create a pipelined multiplier circuit (a * b) using PyRTL's built-in operators.
    
    Args:
        data_type: The data type to use (only used for bitwidth)
        
    Returns:
        The PyRTL working block and the result wire
    """
    # Clear any existing PyRTL design
    pyrtl.reset_working_block()
    
    # Create input and output wires
    a = Input(data_type.bitwidth(), 'a')
    b = Input(data_type.bitwidth(), 'b')
    result = Output(data_type.bitwidth(), 'result')
    
    # Create pipeline registers
    a_reg = pyrtl.Register(bitwidth=data_type.bitwidth(), name='a_reg')
    b_reg = pyrtl.Register(bitwidth=data_type.bitwidth(), name='b_reg')
    
    # Connect input to registers
    a_reg.next <<= a
    b_reg.next <<= b
    
    # Perform multiplication in the next stage using PyRTL's built-in multiplication
    mult_result = a_reg * b_reg
    
    # Truncate the result to match the input bitwidth
    truncated_result = mult_result[:data_type.bitwidth()]
    
    # Connect to output
    result <<= truncated_result
    
    return pyrtl.working_block(), result


def simulate_circuit(block, data_type, num_cycles=10):
    """
    Simulate a circuit with random inputs.
    
    Args:
        block: The PyRTL block to simulate
        data_type: The data type used (only for bitwidth)
        num_cycles: Number of simulation cycles
        
    Returns:
        sim: The simulation object
        trace: The simulation trace
    """
    # Create simulation with tracing enabled
    sim = Simulation()
    
    # Create test data (random integers within the valid range for the bitwidth)
    bitwidth = data_type.bitwidth()
    
    # Handle large bitwidths safely
    if bitwidth > 30:  # Avoid overflow for large bitwidths
        a_values = [np.random.randint(0, 2**16) for _ in range(num_cycles)]
        b_values = [np.random.randint(0, 2**16) for _ in range(num_cycles)]
    else:
        max_val = 2**bitwidth - 1
        a_values = [np.random.randint(0, max_val) for _ in range(num_cycles)]
        b_values = [np.random.randint(0, max_val) for _ in range(num_cycles)]
    
    # Create input dictionaries for each cycle
    input_vectors = []
    for i in range(num_cycles):
        cycle_inputs = {
            'a': a_values[i],
            'b': b_values[i]
        }
        input_vectors.append(cycle_inputs)
    
    # Run simulation for each cycle
    for i in range(num_cycles):
        sim.step(input_vectors[i])
    
    # Get trace from the tracer
    tracer = sim.tracer
    
    return sim, tracer


def main():
    """Main function to create and test the simplified circuits."""
    # Data types to test (only used for bitwidth)
    data_types = [Float8, BF16, Float16, Float32]
    
    # Results dictionary
    results = {
        "simple_adder": {},
        "simple_multiplier": {},
        "pipelined_adder": {},
        "pipelined_multiplier": {}
    }
    
    # Test simple adders
    print("=== Testing Simple Adders ===")
    for dtype in data_types:
        print(f"\nCreating and simulating {dtype.__name__} adder (bitwidth: {dtype.bitwidth()})...")
        block = create_simple_adder(dtype)
        sim, tracer = simulate_circuit(block, dtype)
        
        # Print some results
        if 'result' in tracer.trace:
            output_values = tracer.trace['result']
            print(f"  result: {output_values}")
        
        results["simple_adder"][dtype.__name__] = block
    
    # Test simple multipliers
    print("\n=== Testing Simple Multipliers ===")
    for dtype in data_types:
        print(f"\nCreating and simulating {dtype.__name__} multiplier (bitwidth: {dtype.bitwidth()})...")
        block = create_simple_multiplier(dtype)
        sim, tracer = simulate_circuit(block, dtype)
        
        # Print some results
        if 'result' in tracer.trace:
            output_values = tracer.trace['result']
            print(f"  result: {output_values}")
        
        results["simple_multiplier"][dtype.__name__] = block
    
    # Test pipelined adders
    print("\n=== Testing Pipelined Adders ===")
    for dtype in data_types:
        print(f"\nCreating and simulating {dtype.__name__} pipelined adder (bitwidth: {dtype.bitwidth()})...")
        block, _ = create_pipelined_adder(dtype)
        sim, tracer = simulate_circuit(block, dtype)
        
        # Print some results
        if 'result' in tracer.trace:
            output_values = tracer.trace['result']
            print(f"  result: {output_values}")
        
        results["pipelined_adder"][dtype.__name__] = block
    
    # Test pipelined multipliers
    print("\n=== Testing Pipelined Multipliers ===")
    for dtype in data_types:
        print(f"\nCreating and simulating {dtype.__name__} pipelined multiplier (bitwidth: {dtype.bitwidth()})...")
        block, _ = create_pipelined_multiplier(dtype)
        sim, tracer = simulate_circuit(block, dtype)
        
        # Print some results
        if 'result' in tracer.trace:
            output_values = tracer.trace['result']
            print(f"  result: {output_values}")
        
        results["pipelined_multiplier"][dtype.__name__] = block
    
    return results


if __name__ == "__main__":
    main() 