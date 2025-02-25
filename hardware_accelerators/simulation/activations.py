from typing import Type, Union, Optional
import numpy as np
import pyrtl
from traitlets import default
from hardware_accelerators.dtypes import BaseFloat, BF16
from hardware_accelerators.rtllib.activations import ReluUnit
from .matrix_utils import convert_array_dtype


class ReluSimulator:
    def __init__(self, dtype: Type[BaseFloat] = BF16):
        """Initialize ReLU simulator

        Args:
            dtype: Number format to use (e.g. BF16)
        """
        self.dtype = dtype

    def _setup(self, size: int):
        """Setup PyRTL simulation with given size

        Args:
            size: Number of parallel inputs to process
        """
        pyrtl.reset_working_block()

        # Create input wires
        self.inputs = [
            pyrtl.Input(self.dtype.bitwidth(), f"in_{i}") for i in range(size)
        ]
        self.enable = pyrtl.Input(1, "enable")
        self.start = pyrtl.Input(1, "start")
        self.valid = pyrtl.Input(1, "valid")

        # Create output wires
        self.outputs = [
            pyrtl.Output(self.dtype.bitwidth(), f"out_{i}") for i in range(size)
        ]

        # Create ReLU unit
        self.relu = ReluUnit(size, self.dtype)

        # Connect inputs and outputs
        self.relu.connect_inputs(
            inputs=self.inputs, enable=self.enable, valid=self.valid, start=self.start
        )
        self.relu.connect_outputs(self.outputs)  # type: ignore

        # Create simulations
        self.sim = pyrtl.Simulation()
        self.output_trace = []

    def step(self, **inputs):
        """Step the simulation by one cycle"""
        default_inputs = {
            "enable": 1,
            "start": 1,
            "valid": 1,
            **{f"in_{i}": 0 for i in range(len(self.inputs))},
        }
        default_inputs.update(inputs)
        self.sim.step(default_inputs)
        if self.sim.inspect(self.relu.outputs_valid.name):
            self.output_trace.append(self.relu.inspect_outputs(self.sim))

    def activate(self, data: np.ndarray, enable: bool = True) -> np.ndarray:
        """Apply ReLU activation to input data

        Args:
            data: Input values as numpy array
            enable: Whether to apply ReLU (True) or passthrough (False)

        Returns:
            Activated values in same format as input
        """
        if not isinstance(data, np.ndarray):
            data = np.array(data)

        # Convert input data to binary representation
        binary_data = convert_array_dtype(data, self.dtype)

        # Setup simulation for row size
        if len(data.shape) == 1:
            row_size = len(data)
            binary_data = binary_data.reshape(1, -1)
        else:
            row_size = data.shape[1]

        self._setup(row_size)

        results = []

        # First cycle: Assert start and enable signals
        sim_inputs = {"enable": 1 if enable else 0, "start": 1, "valid": 0}
        for i in range(row_size):
            sim_inputs[f"in_{i}"] = 0
        self.step(**sim_inputs)

        # Process data row by row
        sim_inputs["start"] = 0  # Clear start signal after first cycle
        sim_inputs["valid"] = 1  # Set valid for processing

        for row in binary_data:
            # Load row data
            for i, val in enumerate(row):
                sim_inputs[f"in_{i}"] = val

            # Run simulation step
            self.step(**sim_inputs)

        self.step()
        # Convert results back to numpy array
        results = np.array(self.output_trace)

        # If input was 1D, return 1D result
        if len(data.shape) == 1:
            results = results[0]

        return results
