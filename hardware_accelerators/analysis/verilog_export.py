import os
import sys
import pyrtl
from pyrtl import WireVector, Input, Output, Simulation

# Add the parent directory to the path so we can import from hardware_accelerators
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from hardware_accelerators.dtypes import Float16, Float32, Float8, BF16
from hardware_accelerators.analysis.simple_circuits import (
    create_simple_adder,
    create_simple_multiplier,
    create_pipelined_adder,
    create_pipelined_multiplier,
)


def export_to_verilog(
    block, output_filename, add_reset=True, initialize_registers=False
):
    """
    Export a PyRTL block to a Verilog file.

    Args:
        block: The PyRTL block to export
        output_filename: The filename to write the Verilog code to
        add_reset: If reset logic should be added. Options are:
                  False (no reset logic),
                  True (synchronous reset logic),
                  'asynchronous' (asynchronous reset logic)
        initialize_registers: Initialize Verilog registers to their reset_value
    """
    # Create the output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)

    # Export the block to Verilog
    with open(output_filename, "w") as f:
        pyrtl.output_to_verilog(
            f,
            add_reset=add_reset,
            initialize_registers=initialize_registers,
            block=block,
        )

    print(f"Exported Verilog to {output_filename}")


def export_all_circuits():
    """
    Export all simple circuits to Verilog files.
    """
    # Create output directory
    output_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "verilog_output"
    )
    os.makedirs(output_dir, exist_ok=True)

    # List of data types to use
    data_types = [Float8, Float16, BF16, Float32]

    # Export simple adder for each data type
    for dtype in data_types:
        block = create_simple_adder(dtype)
        output_filename = os.path.join(output_dir, f"simple_adder_{dtype.__name__}.v")
        export_to_verilog(block, output_filename)

    # Export simple multiplier for each data type
    for dtype in data_types:
        block = create_simple_multiplier(dtype)
        output_filename = os.path.join(
            output_dir, f"simple_multiplier_{dtype.__name__}.v"
        )
        export_to_verilog(block, output_filename)

    # Export pipelined adder for each data type
    for dtype in data_types:
        block, _ = create_pipelined_adder(dtype)
        output_filename = os.path.join(
            output_dir, f"pipelined_adder_{dtype.__name__}.v"
        )
        export_to_verilog(block, output_filename, initialize_registers=True)

    # Export pipelined multiplier for each data type
    for dtype in data_types:
        block, _ = create_pipelined_multiplier(dtype)
        output_filename = os.path.join(
            output_dir, f"pipelined_multiplier_{dtype.__name__}.v"
        )
        export_to_verilog(block, output_filename, initialize_registers=True)


def main():
    # Export all circuits
    export_all_circuits()

    print("All circuits exported to Verilog successfully!")


if __name__ == "__main__":
    main()
