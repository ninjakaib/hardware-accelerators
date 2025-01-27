import copy
from pyrtl import WireVector, Input, Output, Simulation, reset_working_block
import numpy as np
from dataclasses import dataclass
from typing import Any, List, Callable, Optional, Type, Dict, Tuple

from .utils import *
from ..rtllib import *
from ..dtypes import *


@dataclass
class SimulationState:
    """Stores the state of the systolic array at a given simulation step"""

    inputs: dict[str, Any]
    weights: np.ndarray
    data: np.ndarray
    outputs: np.ndarray
    accumulators: np.ndarray
    step: int

    def __repr__(self) -> str:
        """Pretty print the simulation state at this step"""
        width = 40
        sep = "-" * width

        return (
            f"\nSimulation State - Step {self.step}\n{sep}\n"
            f"Inputs:\n"
            f"  w_en: {self.inputs['w_en']}\n"
            f"  enable: {self.inputs['enable']}\n"
            f"  weights: {np.array2string(self.inputs['weights'], precision=4, suppress_small=True)}\n"
            f"  data: {np.array2string(self.inputs['data'], precision=4, suppress_small=True)}\n"
            f"\nWeights Matrix:\n{np.array2string(self.weights, precision=4, suppress_small=True)}\n"
            f"\nData Matrix:\n{np.array2string(self.data, precision=4, suppress_small=True)}\n"
            f"\nAccumulators:\n{np.array2string(self.accumulators, precision=4, suppress_small=True)}\n"
            f"\nOutputs:\n{np.array2string(self.outputs, precision=4, suppress_small=True)}\n"
            f"{sep}\n"
        )


class SystolicArraySimulator:
    def __init__(
        self,
        size: int,
        data_type: Type[BaseFloat] = BF16,
        accum_type: Type[BaseFloat] = BF16,
        multiplier: Callable[
            [WireVector, WireVector, Type[BaseFloat]], WireVector
        ] = lmul_fast,
        adder: Callable[
            [WireVector, WireVector, Type[BaseFloat]], WireVector
        ] = float_adder,
        pipeline: bool = False,
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
        self.accum_type = accum_type
        self.pipeline = pipeline
        self.dwidth = data_type.bitwidth()
        self.accwidth = accum_type.bitwidth()
        self.multiplier = multiplier
        self.adder = adder
        self.history: List[SimulationState] = []

    def _setup(self):
        # Setup PyRTL simulation
        reset_working_block()

        # Initialize hardware
        self.array = SystolicArrayDiP(
            size=self.size,
            data_type=self.dtype,
            accum_type=self.accum_type,
            multiplier=self.multiplier,
            adder=self.adder,
            pipeline=self.pipeline,
        )

        self.w_en = self.array.connect_weight_enable(Input(1, "w_en"))
        self.enable = self.array.connect_enable_input(Input(1, "enable"))

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

    @classmethod
    def matrix_multiply(
        cls,
        weights: np.ndarray,
        activations: np.ndarray,
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

        return sim.simulate(weights, activations)

    def simulate(self, weights: np.ndarray, activations: np.ndarray) -> np.ndarray:
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

        # Collect results
        results = []
        for _ in range(self.size + int(self.pipeline)):
            self._step()
            results.insert(0, self.array.inspect_outputs(self.sim, False))

        return np.array(results)

    def _reset_inputs(self):
        """Reset all simulation inputs to 0"""
        for k in self.sim_inputs:
            self.sim_inputs[k] = 0

    def _step(self):
        """Advance simulation one step and record state"""
        self.sim.step(self.sim_inputs)

        # Record simulation state
        state = SimulationState(
            inputs=self.get_readable_inputs(),
            weights=self.array.inspect_weights(self.sim, False),
            data=self.array.inspect_data(self.sim, False),
            outputs=self.array.inspect_outputs(self.sim, False),
            accumulators=self.array.inspect_accumulators(self.sim, False),
            step=len(self.history),
        )
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


# class MatrixMultiplierSimulator:
#     """Simulation environment for systolic array matrix multiplication.

#     Handles initialization of simulation inputs/outputs, matrix loading, stepping through
#     computation cycles, and inspection of internal hardware state. Provides both
#     iterative access to simulation steps and convenience methods for complete matrix
#     multiplication.

#     The simulator tracks the complete output matrix as it's computed row by row,
#     maintaining proper ordering of results as they emerge from the systolic array.

#     Key Methods:
#         set_matrices(): Load input matrices for multiplication
#         calculate(): Run complete simulation and return result
#         matmul(): One-step matrix multiplication (combines set_matrices and calculate)
#         inspect_pe_array(): Get current state of all PE registers
#         inspect_systolic_setup(): Visualize state of input delay network

#     Key Attributes:
#         size: Dimension of the systolic array
#         data_type: Number format for input data/weights
#         accum_type: Number format for accumulation
#         result_matrix: Current state of computed output matrix

#     Example:
#         >>> # Create hardware and simulator
#         >>> mmu = MatrixMultiplier(size=3, data_type=BF16, accum_type=BF16,
#         ...                       multiplier_type=lmul_fast, adder_type=float_adder)
#         >>> sim = MatrixMultiplierSimulator(mmu)
#         >>>
#         >>> # Option 1: Quick matrix multiply
#         >>> A = np.array([[1,2,3], [4,5,6], [7,8,9]])
#         >>> B = np.eye(3)
#         >>> result = sim.matmul(A, B)
#         >>>
#         >>> # Option 2: Step-by-step simulation
#         >>> sim.set_matrices(A, B)
#         >>> for step, row_result in enumerate(sim):
#         ...     print(f"Step {step}:")
#         ...     print(sim.inspect_pe_array())  # View internal state
#         ...     print(f"Output row: {row_result}")
#         >>> final_result = sim.result_matrix
#     """

#     def __init__(self, matrix_multiplier: MatrixMultiplier):
#         """Initialize simulation environment for a matrix multiplier.

#         Args:
#             matrix_multiplier: Instance of MatrixMultiplier hardware to simulate

#         Creates input/output ports and connects them to the matrix multiplier hardware.
#         Initializes simulation state including the result tracking matrix.
#         """

#         self.mmu = matrix_multiplier
#         self.size = matrix_multiplier.size
#         self.data_type = matrix_multiplier.data_type
#         self.accum_type = matrix_multiplier.accum_type

#         # Create I/O ports
#         self.weight_enable = Input(1, "weight_enable")
#         self.weight_ports = [
#             Input(self.mmu.data_width, f"weight_{i}") for i in range(self.size)
#         ]
#         self.data_ports = [
#             Input(self.mmu.data_width, f"data_{i}") for i in range(self.size)
#         ]
#         self.result_ports = [
#             Output(self.mmu.accum_width, f"result_{i}") for i in range(self.size)
#         ]

#         # Connect ports to matrix multiplier
#         self.mmu.connect_weight_enable(self.weight_enable)
#         self.mmu.connect_weights(self.weight_ports)
#         self.mmu.connect_data(self.data_ports)
#         self.mmu.connect_results(self.result_ports)

#         # Initialize simulation
#         self.sim = Simulation()
#         self.sim_inputs = {
#             "weight_enable": 0,
#             **{f"weight_{i}": 0 for i in range(self.size)},
#             **{f"data_{i}": 0 for i in range(self.size)},
#         }

#         # Initialize matrices as None
#         self.matrix_a = None
#         self.matrix_b = None
#         self._iter_state = None
#         self.result_matrix = np.zeros((self.size, self.size))

#     def set_matrices(self, matrix_a: np.ndarray, matrix_b: np.ndarray):
#         """Load input matrices and prepare simulation state.

#         Args:
#             matrix_a: Input activation matrix of shape (size, size)
#             matrix_b: Weight matrix of shape (size, size)

#         Raises:
#             AssertionError: If matrix dimensions don't match systolic array size
#         """
#         # Verify dimensions
#         assert (
#             matrix_a.shape == matrix_b.shape == (self.size, self.size)
#         ), f"Matrices must be {self.size}x{self.size}"

#         # Convert matrices to specified datatype
#         self.matrix_a = self._convert_matrix(matrix_a)
#         self.matrix_b = self._convert_matrix(matrix_b)

#         # Load weights into PEs
#         self._load_weights()

#     def calculate(self) -> np.ndarray:
#         """Run complete matrix multiplication simulation.

#         Returns:
#             Computed result matrix of shape (size, size)
#         """
#         while next(self):
#             continue
#         return self.result_matrix

#     def matmul(self, matrix_a: np.ndarray, matrix_b: np.ndarray) -> np.ndarray:
#         """Convenience method to perform complete matrix multiplication.

#         Args:
#             matrix_a: Input activation matrix of shape (size, size)
#             matrix_b: Weight matrix of shape (size, size)

#         Returns:
#             Computed result matrix of shape (size, size)
#         """
#         self.set_matrices(matrix_a, matrix_b)
#         return self.calculate()

#     def _convert_matrix(self, matrix: np.ndarray) -> List[List[int]]:
#         """Convert numpy matrix to list of binary values in specified datatype"""
#         return [[self.data_type(x).binint for x in row] for row in matrix]

#     def _load_weights(self):
#         """Load weights into processing elements in reverse row order"""
#         for row in reversed(range(self.size)):
#             for col in range(self.size):
#                 self.sim_inputs[f"weight_{col}"] = self.matrix_b[row][col]
#             self.sim_inputs["weight_enable"] = 1
#             self.sim.step(self.sim_inputs)

#         # Reset weight inputs
#         for i in range(self.size):
#             self.sim_inputs[f"weight_{i}"] = 0
#         self.sim_inputs["weight_enable"] = 0
#         self.sim.step(self.sim_inputs)

#     def __iter__(self):
#         """Initialize iterator state"""
#         if self.matrix_a is None or self.matrix_b is None:
#             raise RuntimeError("Matrices must be set before iteration")

#         self._iter_state = {
#             "row": self.size - 1,  # Start from last row
#             "extra_cycles": self.size * 3 - 1,  # Cycles needed to flush results
#             "phase": "input",  # 'input' or 'flush' phase
#         }
#         return self

#     def __next__(self):
#         """Return next simulation step results"""
#         if self._iter_state is None:
#             raise RuntimeError("Iterator not initialized")

#         # If we're done with both phases, stop iteration
#         if (
#             self._iter_state["phase"] == "flush"
#             and self._iter_state["extra_cycles"] == 0
#         ):
#             raise StopIteration

#         # Handle input phase
#         if self._iter_state["phase"] == "input":
#             if self._iter_state["row"] < 0:
#                 # Transition to flush phase
#                 self._iter_state["phase"] = "flush"
#                 # Clear inputs
#                 for i in range(self.size):
#                     self.sim_inputs[f"data_{i}"] = 0
#             else:
#                 # Load next row of input data
#                 for col in range(self.size):
#                     self.sim_inputs[f"data_{col}"] = self.matrix_a[
#                         self._iter_state["row"]
#                     ][col]
#                 self._iter_state["row"] -= 1

#         # Handle flush phase
#         if self._iter_state["phase"] == "flush":
#             self._iter_state["extra_cycles"] -= 1

#         # Step simulation
#         self.sim.step(self.sim_inputs)

#         # Return current results
#         current_outputs = self.get_current_results()

#         # Shift previous results down and insert new results at top
#         self.result_matrix[1:] = self.result_matrix[:-1]
#         self.result_matrix[0] = current_outputs

#         return current_outputs

#     def get_current_results(self) -> List[BaseFloat]:
#         """Get current values from result output ports

#         Returns:
#             List of values currently present on the result output ports.
#             Length will equal systolic array size (one value per column).
#         """
#         return [
#             self.accum_type(binint=self.sim.inspect(f"result_{i}"))
#             for i in range(self.size)
#         ]

#     def inspect_pe_array(self) -> dict[str, np.ndarray]:
#         """Get current state of all registers in the processing element array.

#         Returns:
#             Dictionary containing three matrices of shape (size, size):
#                 'data': Current values in data registers
#                 'weights': Current values in weight registers
#                 'accum': Current values in accumulator registers
#         """
#         # Initialize matrices to store PE values
#         data_matrix = np.zeros((self.size, self.size))
#         weight_matrix = np.zeros((self.size, self.size))
#         accum_matrix = np.zeros((self.size, self.size))

#         # Populate matrices with current PE values
#         for row in range(self.size):
#             for col in range(self.size):
#                 pe = self.mmu.systolic_array.pe_array[row][col]

#                 # Convert binary values to float using appropriate data types
#                 data_matrix[row, col] = self.data_type(
#                     binint=self.sim.inspect(pe.outputs.data.name)
#                 )
#                 weight_matrix[row, col] = self.data_type(
#                     binint=self.sim.inspect(pe.outputs.weight.name)
#                 )
#                 accum_matrix[row, col] = self.accum_type(
#                     binint=self.sim.inspect(pe.outputs.accum.name)
#                 )

#         return {"data": data_matrix, "weights": weight_matrix, "accum": accum_matrix}

#     def inspect_systolic_setup(self) -> str:
#         """Get formatted string showing state of input delay network.

#         Returns:
#             Multi-line string showing current values in all delay registers,
#             formatted to visualize the diagonal buffering pattern
#         """
#         repr_str = ""
#         for row in range(self.size):
#             input_val = self.data_type(binint=self.sim.inspect(f"data_{row}"))
#             repr_str += f"(input={input_val}) => "

#             for reg in self.mmu.systolic_setup.delay_regs[row]:
#                 val = self.data_type(binint=self.sim.inspect(reg.name))
#                 repr_str += f"{val} -> "
#             repr_str += "\n"
#         return repr_str
