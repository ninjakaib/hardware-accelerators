import copy
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import numpy as np
from pyrtl import Input, Output, Simulation, WireVector, reset_working_block

from ..rtllib.systolic import SystolicArraySimState

from ..dtypes import *
from ..rtllib import *
from .utils import *
from .matrix_utils import pad_and_reshape_vector


# @dataclass
# class SimulationState:
#     """Stores the state of the systolic array at a given simulation step"""

#     inputs: dict[str, Any]
#     weights: np.ndarray
#     data: np.ndarray
#     outputs: np.ndarray
#     accumulators: np.ndarray
#     control_regs: dict
#     step: int | None = None

#     def __repr__(self) -> str:
#         """Pretty print the simulation state at this step"""
#         width = 40
#         sep = "-" * width
#         step_str = f"\nSimulation State - Step {self.step}\n{sep}\n" if self.step is not None else ""

#         return (
#             f"{step_str}"
#             f"Inputs:\n"
#             f"  w_en: {self.inputs['w_en']}\n"
#             f"  enable: {self.inputs['enable']}\n"
#             f"  weights: {np.array2string(self.inputs['weights'], precision=4, suppress_small=True)}\n"
#             f"  data: {np.array2string(self.inputs['data'], precision=4, suppress_small=True)}\n"
#             f"\nWeights Matrix:\n{np.array2string(self.weights, precision=4, suppress_small=True)}\n"
#             f"\nData Matrix:\n{np.array2string(self.data, precision=4, suppress_small=True)}\n"
#             f"\nAccumulators:\n{np.array2string(self.accumulators, precision=4, suppress_small=True)}\n"
#             f"\nControl Registers:\n{self.control_regs}\n"
#             f"\nOutputs:\n{np.array2string(self.outputs, precision=4, suppress_small=True)}\n"
#             f"{sep}\n"
#         )


class SystolicArraySimulator:
    def __init__(
        self,
        size: int,
        data_type: Type[BaseFloat] = BF16,
        weight_type: Type[BaseFloat] = BF16,
        accum_type: Type[BaseFloat] = BF16,
        multiplier: Callable[
            [WireVector, WireVector, Type[BaseFloat]], WireVector
        ] = float_multiplier,
        adder: Callable[
            [WireVector, WireVector, Type[BaseFloat]], WireVector
        ] = float_adder,
        pipeline: bool = False,
        accum_addr_width: int | None = None,
    ):
        """Initialize systolic array simulator

        Args:
            size: Dimension of systolic array (NxN)
            dtype: Number format to use (e.g. BF16)
            pipeline: Whether to use pipelined PEs
            multiplier: Multiplication implementation
            adder: Addition implementation
        """
        self.size = size
        self.dtype = data_type
        self.wtype = weight_type
        self.accum_type = accum_type
        self.pipeline = pipeline
        self.dwidth = data_type.bitwidth()
        self.wwidth = weight_type.bitwidth()
        self.accwidth = accum_type.bitwidth()
        self.multiplier = multiplier
        self.adder = adder
        self.addr_width = accum_addr_width
        self.history: List[SystolicArraySimState] = []

    def _setup(self):
        # Setup PyRTL simulation
        reset_working_block()

        # Initialize hardware
        self.array = SystolicArrayDiP(
            size=self.size,
            data_type=self.dtype,
            weight_type=self.wtype,
            accum_type=self.accum_type,
            multiplier=self.multiplier,
            adder=self.adder,
            pipeline=self.pipeline,
            accum_addr_width=self.addr_width,
        )

        self.w_en = self.array.connect_weight_enable(Input(1, "w_en"))
        self.enable = self.array.connect_enable_input(Input(1, "enable"))
        if self.addr_width is not None:
            self.array.connect_inputs(accum_addr=Input(self.addr_width, "acc_addr"))

        self.w_ins = [Input(self.dwidth, f"weight_{i}") for i in range(self.size)]
        self.d_ins = [Input(self.dwidth, f"data_{i}") for i in range(self.size)]
        self.acc_outs = [Output(self.dwidth, f"result_{i}") for i in range(self.size)]

        for i in range(self.size):
            self.array.connect_weight_input(i, self.w_ins[i])
            self.array.connect_data_input(i, self.d_ins[i])
            self.array.connect_result_output(i, self.acc_outs[i])

        self.sim = Simulation()
        self.history = []
        self.sim_inputs = {
            w.name: 0 for w in [self.w_en, self.enable, *self.w_ins, *self.d_ins]
        }
        if self.addr_width is not None:
            self.sim_inputs["acc_addr"] = 0

    @classmethod
    def matrix_multiply(
        cls,
        activations: np.ndarray,
        weights: np.ndarray,
        dtype: Optional[Type[BaseFloat]] = None,
    ) -> np.ndarray:
        """Perform matrix multiplication using systolic array

        Args:
            weights: Weight matrix
            activations: Activation matrix
            dtype: Optional number format override
            pipeline: Whether to use pipelined PEs

        Returns:
            Tuple of (result matrix, simulation history)
        """
        # Validate inputs
        if weights.shape != activations.shape:
            raise ValueError("Weight and activation matrices must have same shape")
        if weights.shape[0] != weights.shape[1]:
            raise ValueError("Only square matrices supported")

        # Create simulator instance
        size = weights.shape[0]
        dtype = dtype or BF16
        sim = cls(size=size, data_type=dtype)

        return sim.simulate(activations=activations, weights=weights)

    def simulate(self, activations: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Instance method to run simulation

        Args:
            weights: Weight matrix
            activations: Activation matrix

        Returns:
            Tuple of (result matrix, simulation history)
        """
        # Validate inputs
        if weights.shape != (self.size, self.size) or activations.shape != (
            self.size,
            self.size,
        ):
            raise ValueError(f"Input matrices must be {self.size}x{self.size}")

        # Convert and permutate matrices
        weights = convert_array_dtype(permutate_weight_matrix(weights), self.dtype)
        activations = convert_array_dtype(activations, self.dtype)

        self._setup()

        # Load weights except top row
        self.sim_inputs["w_en"] = 1
        for row in range(self.size - 1):
            for col in range(self.size):
                self.sim_inputs[self.w_ins[col].name] = weights[-row - 1][col]
            self._step()

        # Load top row weights and first data row
        self.sim_inputs["enable"] = 1
        for col in range(self.size):
            self.sim_inputs[self.w_ins[col].name] = weights[0][col]
            self.sim_inputs[self.d_ins[col].name] = activations[-1][col]
        self._step()

        # Disable weight loading
        self.sim_inputs["w_en"] = 0

        # Feed remaining data
        for row in range(1, self.size):
            for col in range(self.size):
                self.sim_inputs[self.d_ins[col].name] = activations[-row - 1][col]
            self._step()

        # Additional step to flush pipeline
        self._step()
        self._reset_inputs()
        if self.pipeline:
            self._step()

        # Collect results
        results = []
        # while self.sim.inspect(self.array.control_out.name):
        #     results.insert(0, self.array.inspect_outputs(self.sim, False))
        #     self._step()
        for _ in range(self.size):  # + int(self.pipeline)):
            self._step()
            results.insert(0, self.array.inspect_outputs(self.sim, False))

        return np.array(results)

    def simulate_vector(
        self, activations: np.ndarray, weights: np.ndarray, addr: int | None = None
    ) -> np.ndarray:
        """Instance method to run simulation

        Args:
            weights: Weight matrix
            activations: Activation matrix

        Returns:
            Tuple of (result matrix, simulation history)
        """
        # Convert and permutate matrices
        weights = convert_array_dtype(permutate_weight_matrix(weights), self.dtype)
        activations = convert_array_dtype(activations, self.dtype)
        activation_matrix = pad_and_reshape_vector(activations, self.size)

        self._setup()

        # Load weights except top row
        self.sim_inputs["w_en"] = 1
        for row in range(self.size - 1):
            for col in range(self.size):
                self.sim_inputs[self.w_ins[col].name] = weights[-row - 1][col]
            self._step()

        # Load top row weights and first data row
        self.sim_inputs["enable"] = 1
        if addr is not None:
            self.sim_inputs["acc_addr"] = addr
        for col in range(self.size):
            self.sim_inputs[self.w_ins[col].name] = weights[0][col]
            self.sim_inputs[self.d_ins[col].name] = activation_matrix[0][col]
        self._step()

        # Disable all control signals
        self._reset_inputs()

        # Flush the results out
        for _ in range(self.size):
            self._step()

        # Additional step to flush pipeline
        if self.pipeline:
            self._step()

        # Collect results
        # while self.sim.inspect(self.array.control_out.name):
        #     results.insert(0, self.array.inspect_outputs(self.sim, False))
        #     self._step()
        # for _ in range(self.size):  # + int(self.pipeline)):
        self._step()
        results = self.array.inspect_outputs(self.sim, False)

        return np.array(results)

    def _reset_inputs(self):
        """Reset all simulation inputs to 0"""
        for k in self.sim_inputs:
            self.sim_inputs[k] = 0

    def _step(self):
        """Advance simulation one step and record state"""
        self.sim.step(self.sim_inputs)

        # Record simulation state
        state = self.array.get_state(self.sim, len(self.history))
        # state = SimulationState(
        #     inputs=self.get_readable_inputs(),
        #     weights=self.array.inspect_weights(self.sim, False),
        #     data=self.array.inspect_data(self.sim, False),
        #     outputs=self.array.inspect_outputs(self.sim, False),
        #     accumulators=self.array.inspect_accumulators(self.sim, False),
        #     control_regs=self.array.inspect_control_regs(self.sim),
        #     step=len(self.history),
        # )
        self.history.append(state)

    def get_readable_inputs(self) -> dict[str, Any]:
        """Convert binary simulation inputs to human-readable floating point values.

        Returns:
            Dictionary with consolidated inputs:
                - 'w_en', 'enable': Binary control signals
                - 'weights': Array of weight input values in floating point
                - 'data': Array of data input values in floating point
        """
        readable = {
            "w_en": self.sim_inputs["w_en"],
            "enable": self.sim_inputs["enable"],
            "weights": np.zeros(len([k for k in self.sim_inputs if "weight_" in k])),
            "data": np.zeros(len([k for k in self.sim_inputs if "data_" in k])),
        }

        # Convert weight inputs to array
        for i, key in enumerate(sorted([k for k in self.sim_inputs if "weight_" in k])):
            binary = self.sim_inputs[key]
            readable["weights"][i] = self.dtype(binint=binary).decimal_approx

        # Convert data inputs to array
        for i, key in enumerate(sorted([k for k in self.sim_inputs if "data_" in k])):
            binary = self.sim_inputs[key]
            readable["data"][i] = self.dtype(binint=binary).decimal_approx

        return readable

    def __repr__(self) -> str:
        """Detailed representation of simulator configuration and state"""
        config = (
            f"SystolicArrayDiPSimulator(\n"
            f"  size: {self.size}x{self.size}\n"
            f"  data_type: {self.dtype.__name__}\n"
            f"  accum_type: {self.accum_type.__name__}\n"
            f"  pipeline: {self.pipeline}\n"
            f"  steps_simulated: {len(self.history)}\n"
        )

        if not self.history:
            return config + "  history: empty\n)"

        # Add summary of simulation history
        last_state = self.history[-1]
        history = (
            f"  Latest State (Step {last_state.step}):\n"
            f"    Weights Shape: {last_state.weights.shape}\n"
            f"    Data Shape: {last_state.data.shape}\n"
            f"    Latest Outputs: {np.array2string(last_state.outputs, precision=3)}\n"
        )

        return config + history + ")"
