from ..dtypes import *

from typing import Type
from pyrtl import (
    MemBlock,
    WireVector,
    Register,
    conditional_assignment,
    otherwise,
    chop,
)


class BufferMemory:
    def __init__(
        self, array_size: int, data_type: Type[BaseFloat], weight_type: Type[BaseFloat]
    ):
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
