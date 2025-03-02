import pyrtl
from pyrtl import WireVector, Input, Output, Simulation, CompiledSimulation
import numpy as np
import sys
import os

# Add the parent directory to the path so we can import from hardware_accelerators
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from hardware_accelerators.rtllib.accelerator import CompiledAccelerator, AcceleratorConfig
from hardware_accelerators.rtllib.adders import float_adder
from hardware_accelerators.rtllib.multipliers import float_multiplier
from hardware_accelerators.dtypes import Float16, Float32, Float8, BF16


def create_pipeline_multiply_add(data_type, array_size=2, pipeline=True):
    """
    Create a simple pipeline multiply and add accelerator.
    
    Args:
        data_type: The floating-point data type to use
        array_size: Size of the systolic array (default: 2)
        pipeline: Whether to use pipelining (default: True)
        
    Returns:
        accelerator: The compiled accelerator instance
    """
    # Clear any existing PyRTL design
    pyrtl.reset_working_block()
    
    # Create accelerator configuration
    config = AcceleratorConfig(
        array_size=array_size,
        num_weight_tiles=1,
        data_type=data_type,
        weight_type=data_type,
        accum_type=data_type,
        pe_adder=float_adder,
        accum_adder=float_adder,
        pe_multiplier=float_multiplier,
        pipeline=pipeline,
        accum_addr_width=1
    )
    
    # Create the accelerator
    accelerator = CompiledAccelerator(config)
    
    # Create input and output wires
    data_enable = Input(1, 'data_enable')
    data_inputs = [Input(data_type.bitwidth(), f'data_in_{i}') for i in range(array_size)]
    
    weight_enable = Input(1, 'weight_enable')
    weights_in = [Input(data_type.bitwidth(), f'weight_in_{i}') for i in range(array_size)]
    
    accum_addr = Input(1, 'accum_addr')
    accum_mode = Input(1, 'accum_mode')
    
    act_start = Input(1, 'act_start')
    act_func = Input(1, 'act_func')
    
    outputs = [Output(data_type.bitwidth(), f'out_{i}') for i in range(array_size)]
    valid = Output(1, 'valid')
    
    # Connect inputs and outputs
    accelerator.connect_inputs(
        data_enable=data_enable,
        data_inputs=data_inputs,
        weight_enable=weight_enable,
        weights_in=weights_in,
        accum_addr=accum_addr,
        accum_mode=accum_mode,
        act_start=act_start,
        act_func=act_func
    )
    
    accelerator.connect_outputs(outputs, valid)
    
    return accelerator


def simulate_pipeline_multiply_add(accelerator, data_type, array_size=2, num_cycles=10):
    """
    Simulate the pipeline multiply add accelerator.
    
    Args:
        accelerator: The compiled accelerator instance
        data_type: The floating-point data type used
        array_size: Size of the systolic array
        num_cycles: Number of simulation cycles
        
    Returns:
        sim: The simulation object
        trace: The simulation trace
    """
    # Create simulation with tracing enabled
    sim = Simulation()
    
    # Create test data
    data_values = []
    weight_values = []
    
    for i in range(num_cycles):
        # Create random data values
        data_row = [data_type(np.random.uniform(-1.0, 1.0)).binint for _ in range(array_size)]
        weight_row = [data_type(np.random.uniform(-1.0, 1.0)).binint for _ in range(array_size)]
        
        data_values.append(data_row)
        weight_values.append(weight_row)
    
    # Create input dictionaries for each cycle
    input_vectors = []
    for i in range(num_cycles):
        cycle_inputs = {
            'data_enable': 1,
            'weight_enable': 1,
            'accum_addr': 0,
            'accum_mode': 0,
            'act_start': 1,
            'act_func': 0  # No activation function (passthrough)
        }
        
        # Add data and weight inputs
        for j in range(array_size):
            cycle_inputs[f'data_in_{j}'] = data_values[i][j]
            cycle_inputs[f'weight_in_{j}'] = weight_values[i][j]
        
        input_vectors.append(cycle_inputs)
    
    # Run simulation for each cycle
    for i in range(num_cycles):
        sim.step(input_vectors[i])
    
    # Get trace from the tracer
    tracer = sim.tracer
    
    return sim, tracer


def main():
    """Main function to create and test the pipeline multiply add accelerator."""
    # Create accelerators with different data types
    data_types = [Float8, BF16, Float16, Float32]
    accelerators = {}
    
    for dtype in data_types:
        print(f"Creating accelerator with {dtype.__name__}...")
        accelerator = create_pipeline_multiply_add(dtype)
        accelerators[dtype.__name__] = accelerator
        
        # Simulate the accelerator
        sim, tracer = simulate_pipeline_multiply_add(accelerator, dtype)
        
        # Print some results
        print(f"Simulation results for {dtype.__name__}:")
        for i in range(2):  # Just print first two outputs
            wire_name = f'out_{i}'
            if wire_name in tracer.trace:
                output_values = tracer.trace[wire_name]
                print(f"  {wire_name}: {output_values}")
        
        print()
    
    return accelerators


if __name__ == "__main__":
    main() 