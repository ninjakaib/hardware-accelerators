from abc import ABC, abstractmethod
from typing import Callable, List, Type

import numpy as np
from pyrtl import Const, Register, Simulation, WireVector, chop

from hardware_accelerators import *
from hardware_accelerators.simulation import *

from ..dtypes.base import BaseFloat
from .processing_element import ProcessingElement

# TODO: Add float type conversion logic to pass different bitwidths to the accumulator
# TODO: specify different dtypes for weights and activations


class BaseSystolicArray(ABC):
    def __init__(
        self,
        size: int,
        data_type: Type[BaseFloat],
        accum_type: Type[BaseFloat],
        multiplier: Callable[[WireVector, WireVector, Type[BaseFloat]], WireVector],
        adder: Callable[[WireVector, WireVector, Type[BaseFloat]], WireVector],
    ):
        """Base class for implementing systolic array hardware structures

        Args:
            size: N for NxN array
            data_type: Number format for inputs (Float8, BF16)
            accum_type: Number format for accumulation
            multiplier: Multiplier implementation to use
            adder: Adder implementation to use
        """
        # Set configuration attributes
        self.size = size
        self.data_type = data_type
        self.accum_type = accum_type
        data_width = data_type.bitwidth()
        accum_width = accum_type.bitwidth()
        self.multiplier = multiplier
        self.adder = adder

        # Input wires
        self.data_in = [WireVector(data_width) for _ in range(size)]
        self.weights_in = [WireVector(data_width) for _ in range(size)]
        self.results_out = [WireVector(accum_width) for _ in range(size)]

        # Control wires
        self.weight_enable = WireVector(1)

        # Create PE array
        self.pe_array = self._create_pe_array()

        # Connect PEs in systolic pattern based on dataflow type
        self._connect_array()

    @abstractmethod
    def _create_pe_array(self) -> List[List[ProcessingElement]]:
        return [
            [
                ProcessingElement(
                    self.data_type,
                    self.accum_type,
                    self.multiplier,
                    self.adder,
                )
                for _ in range(self.size)
            ]
            for _ in range(self.size)
        ]

    @abstractmethod
    def _connect_array(self):
        pass

    # -----------------------------------------------------------------------------
    # Connection functions return their inputs to allow for more concise simulation
    # -----------------------------------------------------------------------------
    def connect_weight_enable(self, source: WireVector):
        """Connect weight load enable signal"""
        self.weight_enable <<= source
        return source

    def connect_data_input(self, row: int, source: WireVector):
        """Connect data input for specified row"""
        assert 0 <= row < self.size
        self.data_in[row] <<= source
        return source

    def connect_weight_input(self, col: int, source: WireVector):
        """Connect weight input for specified column"""
        assert 0 <= col < self.size
        self.weights_in[col] <<= source
        return source

    def connect_result_output(self, col: int, dest: WireVector):
        """Connect result output from specified column"""
        assert 0 <= col < self.size
        dest <<= self.results_out[col]
        return dest

    # -----------------------------------------------------------------------------
    # Simulation helper methods
    # -----------------------------------------------------------------------------
    def inspect_weights(self, sim: Simulation, verbose: bool = True):
        weights = np.zeros((self.size, self.size))
        enabled = sim.inspect(self.weight_enable.name) == 1
        for row in range(self.size):
            for col in range(self.size):
                w = sim.inspect(self.pe_array[row][col].weight_reg.name)
                weights[row][col] = self.data_type(binint=w).decimal_approx
        if verbose:
            print(f"Weights: {enabled=}")
            print(np.array_str(weights, precision=3, suppress_small=True), "\n")
        return weights

    def inspect_data(self, sim: Simulation, verbose: bool = True):
        data = np.zeros((self.size, self.size))
        for row in range(self.size):
            for col in range(self.size):
                d = sim.inspect(self.pe_array[row][col].data_reg.name)
                data[row][col] = self.data_type(binint=d).decimal_approx
        if verbose:
            print("Data:")
            print(np.array_str(data, precision=4, suppress_small=True), "\n")
        return data

    def inspect_accumulators(self, sim: Simulation, verbose: bool = True):
        acc_regs = np.zeros((self.size, self.size))
        for row in range(self.size):
            for col in range(self.size):
                d = sim.inspect(self.pe_array[row][col].accum_reg.name)
                acc_regs[row][col] = self.accum_type(binint=d).decimal_approx
        if verbose:
            print("Data:")
            print(np.array_str(acc_regs, precision=4, suppress_small=True), "\n")
        return acc_regs

    def inspect_outputs(self, sim: Simulation, verbose: bool = True):
        current_results = np.zeros(self.size)
        for i in range(self.size):
            r = sim.inspect(self.results_out[i].name)
            current_results[i] = self.accum_type(binint=r).decimal_approx
        if verbose:
            print("Output:")
            print(np.array_str(current_results, precision=4, suppress_small=True), "\n")
        return current_results


class SystolicArrayDiP(BaseSystolicArray):
    def __init__(
        self,
        size: int,
        data_type: Type[BaseFloat],
        accum_type: Type[BaseFloat],
        multiplier: Callable[[WireVector, WireVector, Type[BaseFloat]], WireVector],
        adder: Callable[[WireVector, WireVector, Type[BaseFloat]], WireVector],
        pipeline: bool = False,
    ):
        """Initialize systolic array hardware structure.

        This class uses Ḏiagonal-I̱nput and P̱ermutated weight-stationary (DiP) dataflow.

        Args:
            size: N for NxN array
            data_type: Number format for inputs (Float8, BF16)
            accum_type: Number format for accumulation
            multiplier: Multiplier implementation to use
            adder: Adder implementation to use
            pipeline: Add pipeline register after multiplication in processing element
        """
        self.pipeline = pipeline

        # Control signal registers to propogate signal down the array
        self.enable_in = WireVector(1)
        self.control_registers = [Register(1) for _ in range(size)]

        # If PEs contain an extra pipeline stage, 1 additional control reg is needed
        if self.pipeline:
            self.control_registers.append(Register(1))

        super().__init__(size, data_type, accum_type, multiplier, adder)

    def _create_pe_array(self) -> List[List[ProcessingElement]]:
        # Create PE array
        return [
            [
                ProcessingElement(
                    self.data_type,
                    self.accum_type,
                    self.multiplier,
                    self.adder,
                    self.pipeline,
                )
                for _ in range(self.size)
            ]
            for _ in range(self.size)
        ]

    def _connect_array(self):
        """Connect processing elements in DiP configuration
        - All data flows top to bottom diagonally shifted
        - Weights are loaded simultaneously across array
        - Data inputs arrive synchronously
        """
        # Connect control signal registers that propagate down the array
        self.control_registers[0].next <<= self.enable_in
        for i in range(1, len(self.control_registers)):
            self.control_registers[i].next <<= self.control_registers[i - 1]

        for row in range(self.size):
            for col in range(self.size):
                pe = self.pe_array[row][col]

                # Connect PE inputs:
                # First row gets external input, others connect to PE above
                if row == 0:
                    pe.connect_data_enable(self.enable_in)
                    pe.connect_data(self.data_in[col])
                    pe.connect_weight(self.weights_in[col])
                    pe.connect_accum(Const(0))

                # DiP config: PEs connected to previous row and data diagonally shifted by 1
                else:
                    pe.connect_data_enable(self.control_registers[row - 1])
                    pe.connect_data(self.pe_array[row - 1][col - self.size + 1])
                    pe.connect_weight(self.pe_array[row - 1][col])
                    pe.connect_accum(self.pe_array[row - 1][col])

                # Delay the control signal for accumulator by num pipeline stages
                if self.pipeline:
                    pe.connect_mul_enable(self.control_registers[row])
                    pe.connect_adder_enable(self.control_registers[row + 1])
                else:
                    pe.connect_adder_enable(self.control_registers[row])

                # Connect weight enable signal (shared by all PEs)
                pe.connect_weight_enable(self.weight_enable)

                # Connect bottom row results to output ports
                if row == self.size - 1:
                    self.results_out[col] <<= pe.outputs.accum

    def connect_enable_input(self, source: WireVector):
        """Connect PE enable signal. Controls writing to the data input register"""
        self.enable_in <<= source
        return source


# TODO: Add control logic
class SystolicArrayWS(BaseSystolicArray):
    """Hardware implementation of a configurable systolic processing array.

    Creates and connects an NxN array of processing elements (PEs) in a systolic pattern.
    Currently implements weight-stationary dataflow where:
    - Input activations flow left to right through the array
    - Weights are loaded and remain stationary in PEs
    - Partial sums flow top to bottom and accumulate

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
    """

    def __init__(
        self,
        size: int,
        data_type: Type[BaseFloat],
        accum_type: Type[BaseFloat],
        multiplier,
        adder,
    ):
        """Initialize systolic array hardware structure for weight stationary dataflow

        Args:
            size: N for NxN array
            data_type: Number format for inputs (Float8, BF16)
            accum_type: Number format for accumulation
            multiplier: Multiplier implementation to use
            adder: Adder implementation to use
            pipeline: Add pipeline register after multiplication
        """

        self.systolic_setup = SystolicSetup(size, self.data_type)
        self.result_buffer = SystolicSetup(size, self.accum_type)

        super().__init__(size, data_type, accum_type, multiplier, adder)

    def _create_pe_array(self) -> List[List[ProcessingElement]]:
        return super()._create_pe_array()

    def _connect_array(self):
        """Connect processing elements in systolic pattern

        Data flow patterns:
        - Activations flow left to right, must be diagonally buffered with FIFO
        - Weights flow top to bottom
        - Partial sums flow top to bottom
        """
        for row in range(self.size):
            for col in range(self.size):
                pe = self.pe_array[row][col]

                # Connect activation input:
                if col == 0:
                    # First column gets external input
                    self.systolic_setup.connect_input(row, self.data_in[row])
                    pe.connect_data(self.systolic_setup.outputs[row])
                else:
                    # PE data comes from PE to the left
                    pe.connect_data(self.pe_array[row][col - 1])

                # Connect weight and accumulator inputs:
                # First row gets external input, others connect to PE above
                if row == 0:
                    pe.connect_weight(self.weights_in[col])
                    pe.connect_accum(Const(0))
                else:
                    pe.connect_weight(self.pe_array[row - 1][col])
                    pe.connect_accum(self.pe_array[row - 1][col])

                # Connect weight enable to all PEs
                pe.connect_weight_enable(self.weight_enable)

                # Connect bottom row results to output ports
                if row == self.size - 1:
                    self.result_buffer.inputs[-col - 1] <<= pe.outputs.accum
                    self.results_out[col] <<= self.result_buffer.outputs[-col - 1]


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

    def __init__(self, size: int, dtype: Type[BaseFloat]):
        """Initialize delay register network

        Args:
            size: Number of rows in network
            data_width: Bit width of data values
        """
        self.size = size
        self.data_width = dtype.bitwidth()

        # Create input wires for each row
        self.inputs = [WireVector(self.data_width) for _ in range(size)]

        # Create delay register network - more delays for lower rows
        self.delay_regs = []
        self.outputs = [WireVector(self.data_width) for _ in range(size)]

        for i in range(size):  # Create num rows equal to the size of systolic array
            row: List[Register] = []
            # Number of buffer registers equals row index for lower triangular config
            for j in range(i + 1):
                row.append(Register(self.data_width))
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
