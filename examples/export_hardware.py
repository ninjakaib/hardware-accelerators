import os
import sys

# Add the project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from hardware_accelerators.rtllib.export import (
    export_accelerator,
    export_systolic_array
)
from hardware_accelerators.rtllib import AcceleratorConfig
from hardware_accelerators.rtllib.adders import float_adder
from hardware_accelerators.rtllib.multipliers import float_multiplier
from hardware_accelerators.dtypes import BF16

def export_example():
    # Example 1: Export full accelerator
    config = AcceleratorConfig(
        array_size=4,  # 4x4 systolic array
        num_weight_tiles=2,  # Support 2 weight tiles
        data_type=BF16,  # BFloat16 for activations
        weight_type=BF16,  # BFloat16 for weights
        accum_type=BF16,  # BFloat16 for accumulators
        pe_adder=float_adder,  # Floating point adder for PEs
        accum_adder=float_adder,  # Floating point adder for accumulators
        pe_multiplier=float_multiplier,  # Floating point multiplier for PEs
        pipeline=False,  # No pipelining
        accum_addr_width=4  # Match array size
    )
    
    # Export full accelerator to Verilog
    export_accelerator(
        config=config,
        output_file="accelerator.v",
        module_name="matrix_accelerator"
    )
    
    # Example 2: Export just the systolic array
    export_systolic_array(
        array_size=8,  # 8x8 systolic array
        data_type=BF16,
        weight_type=BF16,
        accum_type=BF16,
        output_file="systolic_array.v",
        module_name="systolic_array_8x8"
    )

if __name__ == "__main__":
    export_example() 