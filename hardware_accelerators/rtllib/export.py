from typing import Callable, Type, TypeVar, Dict, Any
import pyrtl
from .accelerator import AcceleratorConfig, Accelerator
from .systolic import SystolicArrayDiP
from ..dtypes import BaseFloat
from ..simulation.accelerator import AcceleratorSimulator
from ..simulation.utils import convert_array_dtype
from .adders import float_adder
from .multipliers import float_multiplier

T = TypeVar('T')

def create_io_ports(hardware_class: Type[T], config: AcceleratorConfig) -> Dict[str, Any]:
    """Create input/output ports based on simulator requirements.
    
    Args:
        hardware_class: The hardware class being exported
        config: Configuration object specifying hardware parameters
        
    Returns:
        Dictionary of created input/output ports
    """
    ports = {}
    
    # Check if we're dealing with a class type or an instance
    if isinstance(hardware_class, type):
        hardware_type = hardware_class
    else:
        hardware_type = type(hardware_class)
    
    if hardware_type == Accelerator:
        # Create inputs based on AcceleratorSimulator requirements
        ports.update({
            "data_enable": pyrtl.Input(1, "data_enable"),
            "data_inputs": [
                pyrtl.Input(config.data_type.bitwidth(), f"data_in_{i}")
                for i in range(config.array_size)
            ],
            "weight_start": pyrtl.Input(1, "weight_start"),
            "weight_tile_addr": pyrtl.Input(config.weight_tile_addr_width, "weight_tile_addr"),
            "accum_addr": pyrtl.Input(config.accum_addr_width, "accum_addr"),
            "accum_mode": pyrtl.Input(1, "accum_mode"),
            "act_start": pyrtl.Input(1, "act_start"),
            "act_func": pyrtl.Input(1, "act_func"),
        })
        
        # Create outputs based on accelerator requirements
        ports.update({
            f"output_{i}": pyrtl.Output(config.accum_type.bitwidth(), f"output_{i}")
            for i in range(config.array_size)
        })
        
    elif hardware_type == SystolicArrayDiP:
        # Create systolic array specific ports without clock
        data_width = config.data_type.bitwidth()
        weight_width = config.weight_type.bitwidth()
        accum_width = config.accum_type.bitwidth()
        
        ports.update({
            # Data input ports
            "data_inputs": [pyrtl.Input(data_width, f"data_in_{i}") 
                       for i in range(config.array_size)],
            
            # Weight input ports
            "weight_inputs": [pyrtl.Input(weight_width, f"weight_in_{i}") 
                         for i in range(config.array_size)],
            
            # Control signals (excluding clock)
            "rst": pyrtl.Input(1, "rst"),
            "weight_enable": pyrtl.Input(1, "weight_enable"),
            "enable_input": pyrtl.Input(1, "enable_input"),
            
            # Output ports
            "outputs": [pyrtl.Output(accum_width, f"output_{i}") 
                      for i in range(config.array_size)]
        })
    
    return ports

def export_to_verilog(
    config: AcceleratorConfig,
    hardware_generator: Type[T] | Callable[[AcceleratorConfig], T],
    output_file: str,
    module_name: str = "accelerator"
) -> None:
    """Export a hardware design to Verilog based on a configuration.
    
    Args:
        config: Configuration object specifying hardware parameters
        hardware_generator: Either a hardware class to instantiate or a factory function that takes a config
        output_file: Path to output Verilog file
        module_name: Name of the Verilog module to generate
    """
    # Reset PyRTL working block
    pyrtl.reset_working_block()
    
    # Generate hardware instance first to determine its type
    if isinstance(hardware_generator, type):
        hardware = hardware_generator(config)
        hardware_type = hardware_generator
    else:
        hardware = hardware_generator(config)
        hardware_type = type(hardware)
    
    # Create IO ports based on the hardware type
    ports = create_io_ports(hardware_type, config)
    
    # Connect ports to hardware
    if isinstance(hardware, Accelerator):
        # Separate input ports from output ports
        input_ports = {
            k: v for k, v in ports.items() 
            if not k.startswith('output_')
        }
        hardware.connect_inputs(**input_ports)
        
        # Connect output ports
        for i in range(config.array_size):
            ports[f'output_{i}'] <<= hardware.outputs[i]
            
    elif isinstance(hardware, SystolicArrayDiP):
        # Connect systolic array specific ports (clock is handled internally by PyRTL)
        hardware.connect_inputs(
            data_inputs=ports["data_inputs"],
            weight_inputs=ports["weight_inputs"],
            enable_input=ports["enable_input"],
            weight_enable=ports["weight_enable"]
        )
        
        # Connect output ports
        for i in range(config.array_size):
            ports["outputs"][i] <<= hardware.results_out[i]
    
    # Optimize the design
    pyrtl.optimize()
    
    # Export to Verilog
    with open(output_file, 'w') as f:
        # Set the module name in the working block
        pyrtl.working_block().name = module_name
        pyrtl.output_to_verilog(f, add_reset=False)
    
    print(f"Successfully exported Verilog to {output_file}")
    
    # Print summary of the design
    print("\nDesign Summary:")
    print(f"Number of inputs: {len(pyrtl.working_block().wirevector_subset(pyrtl.Input))}")
    print(f"Number of outputs: {len(pyrtl.working_block().wirevector_subset(pyrtl.Output))}")
    print(f"Number of registers: {len(pyrtl.working_block().wirevector_subset(pyrtl.Register))}")
    print(f"Total wire count: {len(pyrtl.working_block().wirevector_set)}")

def export_accelerator(
    config: AcceleratorConfig,
    output_file: str,
    module_name: str = "accelerator"
) -> None:
    """Convenience function to export the Accelerator design to Verilog.
    
    Args:
        config: AcceleratorConfig object specifying hardware parameters
        output_file: Path to output Verilog file
        module_name: Name of the Verilog module to generate
    """
    export_to_verilog(config, Accelerator, output_file, module_name)

def export_systolic_array(
    array_size: int,
    data_type: Type[BaseFloat],
    weight_type: Type[BaseFloat],
    accum_type: Type[BaseFloat],
    output_file: str,
    module_name: str = "systolic_array"
) -> None:
    """Convenience function to export just the systolic array to Verilog.
    
    Args:
        array_size: Size of the systolic array (NxN)
        data_type: Data type for input activations
        weight_type: Data type for weights
        accum_type: Data type for accumulators
        output_file: Path to output Verilog file
        module_name: Name of the Verilog module to generate
    """
    # Create minimal config for systolic array
    config = AcceleratorConfig(
        array_size=array_size,
        data_type=data_type,
        weight_type=weight_type,
        accum_type=accum_type,
        num_weight_tiles=1,  # Minimal value since not used
        pe_adder=float_adder,  # Use default adder
        pe_multiplier=float_multiplier,  # Use default multiplier
        accum_adder=float_adder,  # Use default adder
        pipeline=False,
        accum_addr_width=array_size
    )
    
    # Create a factory function that will construct SystolicArrayDiP with all required args
    def systolic_array_factory(config: AcceleratorConfig) -> SystolicArrayDiP:
        return SystolicArrayDiP(
            size=config.array_size,
            data_type=config.data_type,
            weight_type=config.weight_type,
            accum_type=config.accum_type,
            multiplier=config.pe_multiplier,
            adder=config.pe_adder,
            pipeline=config.pipeline
        )
    
    export_to_verilog(config, systolic_array_factory, output_file, module_name) 