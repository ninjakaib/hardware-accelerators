from ..rtllib.systolic import MatrixMultiplier
from pyrtl import Input, Output, Simulation
import numpy as np
from typing import List
from ..dtypes import BaseFloat


class MatrixMultiplierSimulator:
    """Simulation environment for systolic array matrix multiplication.

    Handles initialization of simulation inputs/outputs, matrix loading, stepping through
    computation cycles, and inspection of internal hardware state. Provides both
    iterative access to simulation steps and convenience methods for complete matrix
    multiplication.

    The simulator tracks the complete output matrix as it's computed row by row,
    maintaining proper ordering of results as they emerge from the systolic array.

    Key Methods:
        set_matrices(): Load input matrices for multiplication
        calculate(): Run complete simulation and return result
        matmul(): One-step matrix multiplication (combines set_matrices and calculate)
        inspect_pe_array(): Get current state of all PE registers
        inspect_systolic_setup(): Visualize state of input delay network

    Key Attributes:
        size: Dimension of the systolic array
        data_type: Number format for input data/weights
        accum_type: Number format for accumulation
        result_matrix: Current state of computed output matrix

    Example:
        >>> # Create hardware and simulator
        >>> mmu = MatrixMultiplier(size=3, data_type=BF16, accum_type=BF16,
        ...                       multiplier_type=lmul_fast, adder_type=float_adder)
        >>> sim = MatrixMultiplierSimulator(mmu)
        >>>
        >>> # Option 1: Quick matrix multiply
        >>> A = np.array([[1,2,3], [4,5,6], [7,8,9]])
        >>> B = np.eye(3)
        >>> result = sim.matmul(A, B)
        >>>
        >>> # Option 2: Step-by-step simulation
        >>> sim.set_matrices(A, B)
        >>> for step, row_result in enumerate(sim):
        ...     print(f"Step {step}:")
        ...     print(sim.inspect_pe_array())  # View internal state
        ...     print(f"Output row: {row_result}")
        >>> final_result = sim.result_matrix
    """

    def __init__(self, matrix_multiplier: MatrixMultiplier):
        """Initialize simulation environment for a matrix multiplier.

        Args:
            matrix_multiplier: Instance of MatrixMultiplier hardware to simulate

        Creates input/output ports and connects them to the matrix multiplier hardware.
        Initializes simulation state including the result tracking matrix.
        """

        self.mmu = matrix_multiplier
        self.size = matrix_multiplier.size
        self.data_type = matrix_multiplier.data_type
        self.accum_type = matrix_multiplier.accum_type

        # Create I/O ports
        self.weight_enable = Input(1, "weight_enable")
        self.weight_ports = [
            Input(self.mmu.data_width, f"weight_{i}") for i in range(self.size)
        ]
        self.data_ports = [
            Input(self.mmu.data_width, f"data_{i}") for i in range(self.size)
        ]
        self.result_ports = [
            Output(self.mmu.accum_width, f"result_{i}") for i in range(self.size)
        ]

        # Connect ports to matrix multiplier
        self.mmu.connect_weight_enable(self.weight_enable)
        self.mmu.connect_weights(self.weight_ports)
        self.mmu.connect_data(self.data_ports)
        self.mmu.connect_results(self.result_ports)

        # Initialize simulation
        self.sim = Simulation()
        self.sim_inputs = {
            "weight_enable": 0,
            **{f"weight_{i}": 0 for i in range(self.size)},
            **{f"data_{i}": 0 for i in range(self.size)},
        }

        # Initialize matrices as None
        self.matrix_a = None
        self.matrix_b = None
        self._iter_state = None
        self.result_matrix = np.zeros((self.size, self.size))

    def set_matrices(self, matrix_a: np.ndarray, matrix_b: np.ndarray):
        """Load input matrices and prepare simulation state.

        Args:
            matrix_a: Input activation matrix of shape (size, size)
            matrix_b: Weight matrix of shape (size, size)

        Raises:
            AssertionError: If matrix dimensions don't match systolic array size
        """
        # Verify dimensions
        assert (
            matrix_a.shape == matrix_b.shape == (self.size, self.size)
        ), f"Matrices must be {self.size}x{self.size}"

        # Convert matrices to specified datatype
        self.matrix_a = self._convert_matrix(matrix_a)
        self.matrix_b = self._convert_matrix(matrix_b)

        # Load weights into PEs
        self._load_weights()

    def calculate(self) -> np.ndarray:
        """Run complete matrix multiplication simulation.

        Returns:
            Computed result matrix of shape (size, size)
        """
        while next(self):
            continue
        return self.result_matrix

    def matmul(self, matrix_a: np.ndarray, matrix_b: np.ndarray) -> np.ndarray:
        """Convenience method to perform complete matrix multiplication.

        Args:
            matrix_a: Input activation matrix of shape (size, size)
            matrix_b: Weight matrix of shape (size, size)

        Returns:
            Computed result matrix of shape (size, size)
        """
        self.set_matrices(matrix_a, matrix_b)
        return self.calculate()

    def _convert_matrix(self, matrix: np.ndarray) -> List[List[int]]:
        """Convert numpy matrix to list of binary values in specified datatype"""
        return [[self.data_type(x).binint for x in row] for row in matrix]

    def _load_weights(self):
        """Load weights into processing elements in reverse row order"""
        for row in reversed(range(self.size)):
            for col in range(self.size):
                self.sim_inputs[f"weight_{col}"] = self.matrix_b[row][col]
            self.sim_inputs["weight_enable"] = 1
            self.sim.step(self.sim_inputs)

        # Reset weight inputs
        for i in range(self.size):
            self.sim_inputs[f"weight_{i}"] = 0
        self.sim_inputs["weight_enable"] = 0
        self.sim.step(self.sim_inputs)

    def __iter__(self):
        """Initialize iterator state"""
        if self.matrix_a is None or self.matrix_b is None:
            raise RuntimeError("Matrices must be set before iteration")

        self._iter_state = {
            "row": self.size - 1,  # Start from last row
            "extra_cycles": self.size * 3 - 1,  # Cycles needed to flush results
            "phase": "input",  # 'input' or 'flush' phase
        }
        return self

    def __next__(self):
        """Return next simulation step results"""
        if self._iter_state is None:
            raise RuntimeError("Iterator not initialized")

        # If we're done with both phases, stop iteration
        if (
            self._iter_state["phase"] == "flush"
            and self._iter_state["extra_cycles"] == 0
        ):
            raise StopIteration

        # Handle input phase
        if self._iter_state["phase"] == "input":
            if self._iter_state["row"] < 0:
                # Transition to flush phase
                self._iter_state["phase"] = "flush"
                # Clear inputs
                for i in range(self.size):
                    self.sim_inputs[f"data_{i}"] = 0
            else:
                # Load next row of input data
                for col in range(self.size):
                    self.sim_inputs[f"data_{col}"] = self.matrix_a[
                        self._iter_state["row"]
                    ][col]
                self._iter_state["row"] -= 1

        # Handle flush phase
        if self._iter_state["phase"] == "flush":
            self._iter_state["extra_cycles"] -= 1

        # Step simulation
        self.sim.step(self.sim_inputs)

        # Return current results
        current_outputs = self.get_current_results()

        # Shift previous results down and insert new results at top
        self.result_matrix[1:] = self.result_matrix[:-1]
        self.result_matrix[0] = current_outputs

        return current_outputs

    def get_current_results(self) -> List[BaseFloat]:
        """Get current values from result output ports

        Returns:
            List of values currently present on the result output ports.
            Length will equal systolic array size (one value per column).
        """
        return [
            self.accum_type(binint=self.sim.inspect(f"result_{i}"))
            for i in range(self.size)
        ]

    def inspect_pe_array(self) -> dict[str, np.ndarray]:
        """Get current state of all registers in the processing element array.

        Returns:
            Dictionary containing three matrices of shape (size, size):
                'data': Current values in data registers
                'weights': Current values in weight registers
                'accum': Current values in accumulator registers
        """
        # Initialize matrices to store PE values
        data_matrix = np.zeros((self.size, self.size))
        weight_matrix = np.zeros((self.size, self.size))
        accum_matrix = np.zeros((self.size, self.size))

        # Populate matrices with current PE values
        for row in range(self.size):
            for col in range(self.size):
                pe = self.mmu.systolic_array.pe_array[row][col]

                # Convert binary values to float using appropriate data types
                data_matrix[row, col] = self.data_type(
                    binint=self.sim.inspect(pe.outputs.data.name)
                )
                weight_matrix[row, col] = self.data_type(
                    binint=self.sim.inspect(pe.outputs.weight.name)
                )
                accum_matrix[row, col] = self.accum_type(
                    binint=self.sim.inspect(pe.outputs.accum.name)
                )

        return {"data": data_matrix, "weights": weight_matrix, "accum": accum_matrix}

    def inspect_systolic_setup(self) -> str:
        """Get formatted string showing state of input delay network.

        Returns:
            Multi-line string showing current values in all delay registers,
            formatted to visualize the diagonal buffering pattern
        """
        repr_str = ""
        for row in range(self.size):
            input_val = self.data_type(binint=self.sim.inspect(f"data_{row}"))
            repr_str += f"(input={input_val}) => "

            for reg in self.mmu.systolic_setup.delay_regs[row]:
                val = self.data_type(binint=self.sim.inspect(reg.name))
                repr_str += f"{val} -> "
            repr_str += "\n"
        return repr_str
