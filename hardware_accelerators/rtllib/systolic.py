from typing import List, Type
from dataclasses import dataclass
from pyrtl import WireVector, Register, Const, chop
from .processing_element import ProcessingElement
from ..dtypes.base import BaseFloat


class MatrixMultiplier:
    """Hardware implementation of a systolic array matrix multiplier.

    This class creates and connects the hardware components needed for systolic array
    matrix multiplication, including the processing element array, input delay network
    for data skewing, and output synchronization buffer.

    The hardware supports both individual wire connections and wide vectorized interfaces
    for data, weights and results. All floating point operations use parameterized
    number formats and arithmetic implementations.

    Attributes:
        size: Dimension N of the NxN systolic array
        data_type: Number format class for input data and weights
        accum_type: Number format class for accumulation
        data_width: Bit width of data/weight values
        accum_width: Bit width of accumulator values
    """

    def __init__(
        self,
        size: int,
        data_type: Type[BaseFloat],
        accum_type: Type[BaseFloat],
        multiplier_type,
        adder_type,
        pipeline_mult: bool = False,
    ):
        """Initialize systolic array hardware structure.

        Args:
            size: N for NxN array dimension
            data_type: Number format class for inputs (e.g. Float8, BF16)
            accum_type: Number format class for accumulation
            multiplier_type: Floating point multiplier implementation to use
            adder_type: Floating point adder implementation to use
            pipeline_mult: If True, adds pipeline register after multiplication

        The hardware is constructed but IO ports are not connected - use the connect_*
        methods to attach external signals after initialization.
        """
        self.size = size
        self.data_type = data_type
        self.accum_type = accum_type
        self.data_width = data_type.bitwidth()
        self.accum_width = accum_type.bitwidth()

        # Create hardware components
        self.systolic_array = SystolicArray(
            size, data_type, accum_type, multiplier_type, adder_type, pipeline_mult
        )
        self.systolic_setup = SystolicSetup(size, self.data_width)
        self.result_buffer = SystolicSetup(size, self.accum_width)

        # Connect internal components
        self._connect_internal_components()

    def _connect_internal_components(self):
        """Connect systolic array to input/output buffers"""
        for i in range(self.size):
            self.systolic_array.connect_data_input(i, self.systolic_setup.outputs[i])
            self.systolic_array.connect_result_output(
                i, self.result_buffer.inputs[-i - 1]
            )

    def _validate_wire_list(
        self, wires: List[WireVector], expected_width: int, purpose: str
    ):
        """Validate a list of wires meets requirements"""
        if len(wires) != self.size:
            raise ValueError(f"{purpose} requires {self.size} wires, got {len(wires)}")
        if not all(isinstance(w, WireVector) for w in wires):
            raise TypeError(f"All {purpose} must be WireVector instances")
        if not all(w.bitwidth == expected_width for w in wires):
            raise ValueError(f"All {purpose} must have bitwidth {expected_width}")

    def _split_wide_wire(
        self, wire: WireVector, width_per_slice: int
    ) -> List[WireVector]:
        """Split a wide wire into equal slices"""
        expected_width = width_per_slice * self.size
        if wire.bitwidth != expected_width:
            raise ValueError(
                f"Wide wire must have bitwidth {expected_width}, got {wire.bitwidth}"
            )
        # Use chop instead of manual slicing
        return chop(wire, *([width_per_slice] * self.size))

    def connect_weight_enable(self, enable: WireVector):
        """Connect weight enable signal"""
        if not isinstance(enable, WireVector) or enable.bitwidth != 1:
            raise ValueError("Weight enable must be 1-bit WireVector")
        self.systolic_array.connect_weight_load(enable)

    def connect_weights(self, weights: WireVector | List[WireVector]):
        """Connect weight inputs either as list of wires or single wide wire"""
        if isinstance(weights, list):
            self._validate_wire_list(weights, self.data_width, "weight inputs")
            weight_wires = weights
        else:
            # Split wide wire into individual weight wires
            weight_wires = chop(weights, *([self.data_width] * self.size))

        for i, wire in enumerate(weight_wires):
            self.systolic_array.connect_weight_input(i, wire)

    def connect_data(self, data: WireVector | List[WireVector]):
        """Connect data inputs either as list of wires or single wide wire"""
        if isinstance(data, list):
            self._validate_wire_list(data, self.data_width, "data inputs")
            data_wires = data
        else:
            # Split wide wire into individual data wires
            data_wires = chop(data, *([self.data_width] * self.size))

        for i, wire in enumerate(data_wires):
            self.systolic_setup.connect_input(i, wire)

    def connect_results(self, results: WireVector | List[WireVector]):
        """Connect result outputs either as list of wires or single wide wire"""
        if isinstance(results, list):
            self._validate_wire_list(results, self.accum_width, "result outputs")
            result_wires = results
        else:
            # Split wide wire into individual result wires
            result_wires = chop(results, *([self.accum_width] * self.size))

        for i, wire in enumerate(result_wires):
            self.result_buffer.connect_output(-i - 1, wire)


@dataclass
class SystolicArrayPorts:
    """Container for array I/O ports"""

    # Input ports for each row/column
    data_in: List[WireVector]  # Input activations flowing left->right
    weights_in: List[WireVector]  # Input weights flowing top->bottom
    weight_load: WireVector  # Weight load enable signal (1-bit)
    # Output ports from bottom row
    results_out: List[WireVector]  # Output partial sums


# TODO: add different ways of initializing systolic array for various dataflow patterns (static weight vs. output)
class SystolicArray:
    """Hardware implementation of a configurable systolic processing array.

    Creates and connects an NxN array of processing elements (PEs) in a systolic pattern.
    Currently implements weight-stationary dataflow where:
    - Input activations flow left to right through the array
    - Weights are loaded and remain stationary in PEs
    - Partial sums flow top to bottom and accumulate

    Future implementations will support different dataflow patterns:
    - Output stationary: Partial sums remain in PEs, inputs flow through
    - Input stationary: Activations remain in PEs, weights flow through
    - Time-multiplexed: Multiple dataflow patterns using the same hardware

    The array provides external interfaces for:
    - Data inputs (one per row)
    - Weight inputs (one per column)
    - Weight load enable signal
    - Result outputs (one per column)

    All connections between PEs follow the systolic pattern with proper
    pipelining and synchronization of data movement.

    Attributes:
        size: Dimension N of the NxN array
        data_type: Number format for input data and weights
        accum_type: Number format for accumulation
        pe_array: 2D list containing all processing elements
        ports: Container for all external interface wires
    """

    def __init__(
        self,
        size: int,
        data_type: Type[BaseFloat],
        accum_type: Type[BaseFloat],
        multiplier_type,
        adder_type,
        pipeline_mult: bool = False,
        dip: bool = False,
    ):
        """Initialize systolic array hardware structure for weight stationary dataflow

        Args:
            size: N for NxN array
            data_type: Number format for inputs (Float8, BF16)
            accum_type: Number format for accumulation
            multiplier_type: Multiplier implementation to use
            adder_type: Adder implementation to use
            pipeline_mult: Add pipeline register after multiplication
            dip: Control the dataflow configuration between PEs (False creates standard weight stationary setup, True uses new architecture and requires permutation of weights before loading)
        """
        self.size = size
        self.dip = dip
        self.data_type = data_type
        self.accum_type = accum_type

        # Create interface wires
        data_width = data_type.bitwidth()
        accum_width = accum_type.bitwidth()

        self.data_in = [WireVector(bitwidth=data_width) for _ in range(size)]
        self.weights_in = [WireVector(bitwidth=data_width) for _ in range(size)]
        self.weight_load = WireVector(bitwidth=1)
        self.results_out = [WireVector(bitwidth=accum_width) for _ in range(size)]

        # Create PE array
        self.pe_array = [
            [
                ProcessingElement(
                    data_type, accum_type, multiplier_type, adder_type, pipeline_mult
                )
                for _ in range(size)
            ]
            for _ in range(size)
        ]

        # Connect PEs in systolic pattern
        self._connect_array()

        # Package ports for external access
        self.ports = SystolicArrayPorts(
            data_in=self.data_in,
            weights_in=self.weights_in,
            weight_load=self.weight_load,
            results_out=self.results_out,
        )

    def _connect_array(self):
        """Connect processing elements in systolic pattern

        Data flow patterns:
        - Activations flow left to right
        - Weights flow top to bottom
        - Partial sums flow top to bottom
        """
        for row in range(self.size):
            for col in range(self.size):
                pe = self.pe_array[row][col]

                # Connect activation input:
                # First column gets external input, others connect to PE on left
                if not self.dip:
                    if col == 0:
                        pe.connect_data(self.data_in[row])
                    else:
                        pe.connect_data(self.pe_array[row][col - 1])

                # Connect weight input:
                # First row gets external input, others connect to PE above
                if row == 0:
                    pe.connect_weight(self.weights_in[col])
                    if self.dip:
                        pe.connect_data(self.data_in[col])
                else:
                    pe.connect_weight(self.pe_array[row - 1][col])
                    if self.dip:
                        pe.connect_data(self.pe_array[row - 1][col - self.size + 1])

                # Connect accumulator input:
                # First row starts at 0, others connect to PE above
                if row == 0:
                    pe.connect_accum(Const(0))
                else:
                    pe.connect_accum(self.pe_array[row - 1][col])

                # Connect weight enable to all PEs
                pe.connect_weight_enable(self.weight_load)

                # Connect bottom row results to output ports
                if row == self.size - 1:
                    self.results_out[col] <<= pe.outputs.accum

    def connect_data_input(self, row: int, source: WireVector):
        """Connect data input for specified row"""
        assert 0 <= row < self.size
        self.data_in[row] <<= source

    def connect_weight_input(self, col: int, source: WireVector):
        """Connect weight input for specified column"""
        assert 0 <= col < self.size
        self.weights_in[col] <<= source

    def connect_weight_load(self, source: WireVector):
        """Connect weight load enable signal"""
        self.weight_load <<= source

    def connect_result_output(self, col: int, dest: WireVector):
        """Connect result output from specified column"""
        assert 0 <= col < self.size
        dest <<= self.results_out[col]


class SystolicSetup:
    """Creates diagonal delay pattern for systolic array I/O

    For a 3x3 array, creates following pattern of registers:
    (R = register, -> = connection)

    Row 0:  [R] ------->
    Row 1:  [R]->[R] -->
    Row 2:  [R]->[R]->[R]

    - Each row i contains i+1 registers
    - Input connects to leftmost register
    - Output reads from rightmost register
    - Can be used for both input and output buffering
    """

    def __init__(self, size: int, data_width: int):
        """Initialize delay register network

        Args:
            size: Number of rows in network
            data_width: Bit width of data values
        """
        self.size = size
        self.data_width = data_width

        # Create input wires for each row
        self.inputs = [WireVector(bitwidth=data_width) for _ in range(size)]

        # Create delay register network - more delays for lower rows
        self.delay_regs = []
        self.outputs = [WireVector(bitwidth=data_width) for _ in range(size)]

        for i in range(size):  # Create num rows equal to the size of systolic array
            row: List[Register] = []
            # Number of buffer registers equals row index for lower triangular config
            for j in range(i + 1):
                row.append(Register(bitwidth=data_width))
                if j != 0:
                    # Left most register connects to inputs, others connect to previous reg
                    row[j].next <<= row[j - 1]

            # Connect row input and output
            row[0].next <<= self.inputs[i]
            self.outputs[i] <<= row[-1]
            self.delay_regs.append(row)

    def connect_input(self, row: int, source: WireVector):
        """Connect input for specified row"""
        assert 0 <= row < self.size
        self.inputs[row] <<= source

    def connect_output(self, row, dest: WireVector):
        """Connect final register in a buffer row to an output destination"""
        dest <<= self.outputs[row]
