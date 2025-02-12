from dataclasses import dataclass
from ..dtypes import *

from typing import List, Type
from pyrtl import (
    MemBlock,
    WireVector,
    Register,
    conditional_assignment,
    otherwise,
    chop,
)


@dataclass
class BufferOutputs:
    """Container for buffer memory output wires.

    Attributes:
        datas_out: List of data output wires, one per array column
        weights_out: List of weight output wires, one per array column
        data_valid: Wire indicating valid data on datas_out
        weight_valid: Wire indicating valid weights on weights_out
    """

    datas_out: List[WireVector]
    """List of data output wires, one per array column"""
    weights_out: List[WireVector]
    """List of weight output wires, one per array column"""
    data_valid: WireVector
    """Wire indicating valid data on datas_out"""
    weight_valid: WireVector
    """Wire indicating valid weights on weights_out"""


class BufferMemory:
    """Dual-bank memory buffer for streaming data and weights to a systolic array.

    This class implements a memory buffer with separate banks for both data and weights,
    designed to feed a systolic array for matrix multiplication. It features:

    - Dual memory banks for both data and weights enabling ping-pong buffering
    - Configurable data widths for both data and weight values
    - Controlled streaming of rows to systolic array
    - Status signals for monitoring buffer operations

    The buffer stores each row of the matrix as a single concatenated entry in memory,
    with the bitwidth scaled by the array size. This enables efficient reading of full
    rows during streaming operations.

    Input Control Wires:
        - data_start_in (1 bit): Initiates data streaming operation
        - data_select_in (1 bit): Selects which data memory bank to read from (0 or 1)
        - weight_start_in (1 bit): Initiates weight streaming operation
        - weight_select_in (1 bit): Selects which weight memory bank to read from (0 or 1)

    Output Status Wires:
        - data_load_busy (1 bit): Indicates data streaming is in progress
        - data_load_done (1 bit): Indicates data streaming has completed
        - weight_load_busy (1 bit): Indicates weight streaming is in progress
        - weight_load_done (1 bit): Indicates weight streaming has completed

    Data Output Wires:
        - datas_out: List of data output wires (data_type.bitwidth() each)
        - weights_out: List of weight output wires (weight_type.bitwidth() each)

    Usage Example:
        buffer = BufferMemory(
            array_size=4,
            data_type=BF16,
            weight_type=BF16
        )

        # Connect control signals
        buffer.connect_inputs(
            data_start=control.data_start,
            data_select=control.data_select,
            weight_start=control.weight_start,
            weight_select=control.weight_select
        )

        # Access outputs
        outputs = buffer.get_outputs()
        systolic_array.connect_data_inputs(outputs.datas_out)
        systolic_array.connect_weight_inputs(outputs.weights_out)
    """

    def __init__(
        self, array_size: int, data_type: Type[BaseFloat], weight_type: Type[BaseFloat]
    ):
        """Initialize the buffer memory with specified dimensions and data types.

        Args:
            array_size: Size N of the NxN systolic array this buffer will feed.
                       Determines the number of parallel output wires and memory organization.

            data_type: Float data type for activation/data values (e.g., BF16, Float8).
                      Determines the bitwidth of data storage and output wires.

            weight_type: Float data type for weight values (e.g., BF16, Float8).
                        Determines the bitwidth of weight storage and output wires.

        Memory Organization:
            - Each memory bank contains array_size entries
            - Each entry stores one full row of the matrix
            - Entry bitwidth = type.bitwidth() * array_size

        The class automatically calculates:
            - Address width based on array_size
            - Memory entry width based on data types and array_size
            - Required control register sizes
        """
        # Configuration parameters
        self.array_size = array_size
        self.addr_width = (array_size - 1).bit_length()
        self.d_width = data_type.bitwidth()
        self.w_width = weight_type.bitwidth()
        self.data_mem_width = self.d_width * array_size
        self.weight_mem_width = self.w_width * array_size

        # Memory Banks
        self.data_mems = [
            MemBlock(bitwidth=self.data_mem_width, addrwidth=self.addr_width)
            for _ in range(2)
        ]
        self.weight_mems = [
            MemBlock(bitwidth=self.weight_mem_width, addrwidth=self.addr_width)
            for _ in range(2)
        ]

        # Control Inputs
        self.data_start = WireVector(1)  # Start data streaming
        self.data_bank = WireVector(1)  # Select data memory bank
        self.weight_start = WireVector(1)  # Start weight streaming
        self.weight_bank = WireVector(1)  # Select weight memory bank

        # State Registers
        self.data_active = Register(1)  # Data streaming in progress
        self.data_addr = Register(self.addr_width)
        self.weight_active = Register(1)  # Weight streaming in progress
        self.weight_addr = Register(self.addr_width)

        # Status Outputs
        self.data_valid = WireVector(1)  # Data output is valid
        self.weight_valid = WireVector(1)  # Weight output is valid

        # Data Outputs
        self.datas_out = [WireVector(self.d_width) for _ in range(array_size)]
        self.weights_out = [WireVector(self.w_width) for _ in range(array_size)]

        # Control Logic
        self._implement_control_logic()

    def _implement_control_logic(self):
        """Implement the control and datapath logic."""
        with conditional_assignment:
            # Data streaming control
            with self.data_start & ~self.data_active:
                self.data_active.next |= 1
                self.data_addr.next |= 0

            with self.data_active:
                # Generate valid signal
                self.data_valid |= 1

                # Stream data from selected memory bank
                with self.data_bank == 0:
                    for out, data in zip(
                        self.datas_out,
                        chop(
                            self.data_mems[0][self.data_addr],
                            *[self.d_width] * self.array_size,
                        ),
                    ):
                        out |= data
                with otherwise:
                    for out, data in zip(
                        self.datas_out,
                        chop(
                            self.data_mems[1][self.data_addr],
                            *[self.d_width] * self.array_size,
                        ),
                    ):
                        out |= data

                # Address counter and completion logic
                with self.data_addr == self.array_size - 1:
                    self.data_active.next |= 0
                with otherwise:
                    self.data_addr.next |= self.data_addr + 1

        with conditional_assignment:
            # Weight streaming control (mirror of data control)
            with self.weight_start & ~self.weight_active:
                self.weight_active.next |= 1
                self.weight_addr.next |= 0

            with self.weight_active:
                self.weight_valid |= 1

                with self.weight_bank == 0:
                    for out, weight in zip(
                        self.weights_out,
                        chop(
                            self.weight_mems[0][self.weight_addr],
                            *[self.w_width] * self.array_size,
                        ),
                    ):
                        out |= weight
                with otherwise:
                    for out, weight in zip(
                        self.weights_out,
                        chop(
                            self.weight_mems[1][self.weight_addr],
                            *[self.w_width] * self.array_size,
                        ),
                    ):
                        out |= weight

                with self.weight_addr == self.array_size - 1:
                    self.weight_active.next |= 0
                with otherwise:
                    self.weight_addr.next |= self.weight_addr + 1

    def connect_inputs(self, data_start, data_bank, weight_start, weight_bank):
        """Connect control signals for the buffer memory.

        Args:
            data_start: Start signal for data streaming (1 bit)
            data_bank: Data memory bank selection (1 bit)
            weight_start: Start signal for weight streaming (1 bit)
            weight_bank: Weight memory bank selection (1 bit)
        """
        if data_start is not None:
            assert len(data_start) == 1
            self.data_start <<= data_start

        if data_bank is not None:
            assert len(data_bank) == 1
            self.data_bank <<= data_bank

        if weight_start is not None:
            assert len(weight_start) == 1
            self.weight_start <<= weight_start

        if weight_bank is not None:
            assert len(weight_bank) == 1
            self.weight_bank <<= weight_bank

    def get_outputs(self) -> BufferOutputs:
        """Get all output wires from the buffer memory.

        Returns:
            BufferOutputs containing:
                - datas_out: List of data output wires [array_size]
                - weights_out: List of weight output wires [array_size]
                - data_valid: Indicates valid data on outputs
                - weight_valid: Indicates valid weights on outputs

        The valid signals should be used to enable downstream components:
        - weight_valid connects to systolic array's weight_enable
        - data_valid indicates when data values are ready to be consumed
        """
        return BufferOutputs(
            datas_out=self.datas_out,
            weights_out=self.weights_out,
            data_valid=self.data_valid,
            weight_valid=self.weight_valid,
        )
