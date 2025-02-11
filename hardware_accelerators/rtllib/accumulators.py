from enum import IntEnum
from typing import Callable, Type

from pyrtl import (
    MemBlock,
    Register,
    RomBlock,
    WireVector,
    conditional_assignment,
    otherwise,
)

from ..dtypes.base import BaseFloat


class TiledAccumulatorFSM(IntEnum):
    IDLE = 0
    WRITING = 1


class ReadAccumulatorFSM(IntEnum):
    IDLE = 0
    READING = 1


class TiledAddressGenerator:
    """Hardware control unit for managing tiled memory access patterns.

    Provides dual finite state machines for independent read and write operations,
    with integrated support for accumulate/overwrite modes. Generates addresses
    for accessing data organized in tiles, where each tile contains array_size
    rows of data.

    Features:
    - Separate read/write FSMs for overlapped operation
    - Base address ROM for fast tile address computation
    - Mode control for accumulate vs overwrite operations
    - Row tracking within tiles
    - Status signals for external coordination
    """

    def __init__(self, tile_addr_width: int, array_size: int):
        """Initialize the address generator.

        Args:
            tile_addr_width: Number of bits for addressing tiles. Determines
                number of tiles as 2^tile_addr_width.
            array_size: Number of rows per tile, matching systolic array
                dimension. Also determines address increment pattern.

        The internal address width is computed to accommodate all required
        addresses (num_tiles * array_size locations).
        """
        self.array_size = array_size
        self.num_tiles = 2**tile_addr_width
        self.internal_addr_width = (self.num_tiles * array_size - 1).bit_length()

        # Base address ROM
        base_addrs = [i * array_size for i in range(self.num_tiles)]
        self.base_addr_rom = RomBlock(
            bitwidth=self.internal_addr_width,
            addrwidth=tile_addr_width,
            romdata=base_addrs,
        )

        # ================== Write Interface ==================
        self._tile_addr = WireVector(tile_addr_width)
        self._write_start = WireVector(1)
        self._write_mode = WireVector(1)  # 0=overwrite, 1=accumulate
        self._write_valid = WireVector(1)

        # Write state registers
        self.write_state = Register(1)
        self.write_addr = Register(self.internal_addr_width)
        self.write_row = Register(array_size.bit_length())
        self.write_mode_reg = Register(1)  # Stores mode for current operation

        # Outputs
        self.write_addr_out = WireVector(self.internal_addr_width)
        self.write_enable = WireVector(1)
        self.write_busy = WireVector(1)
        self.write_done = WireVector(1)
        self.write_mode_out = WireVector(1)

        # ================== Read Interface ==================
        self._read_tile_addr = WireVector(tile_addr_width)
        self._read_start = WireVector(1)

        # Read state registers
        self.read_state = Register(1)
        self.read_addr = Register(self.internal_addr_width)
        self.read_row = Register(array_size.bit_length())

        # Outputs
        self.read_addr_out = WireVector(self.internal_addr_width)
        self.read_busy = WireVector(1)
        self.read_done = WireVector(1)

        self._implement_write_fsm()
        self._implement_read_fsm()

    def _implement_write_fsm(self):
        write_base = self.base_addr_rom[self._tile_addr]

        # Combinational outputs
        self.write_addr_out <<= self.write_addr
        self.write_enable <<= (
            self.write_state == TiledAccumulatorFSM.WRITING
        ) & self._write_valid
        self.write_busy <<= self.write_state == TiledAccumulatorFSM.WRITING
        self.write_done <<= (self.write_state == TiledAccumulatorFSM.WRITING) & (
            self.write_row == self.array_size
        )
        self.write_mode_out <<= self.write_mode_reg

        with conditional_assignment:
            # IDLE State
            with self.write_state == TiledAccumulatorFSM.IDLE:
                with self._write_start:
                    self.write_state.next |= TiledAccumulatorFSM.WRITING
                    self.write_addr.next |= write_base
                    self.write_row.next |= 0
                    self.write_mode_reg.next |= self._write_mode  # Latch mode

            # WRITING State
            with self.write_state == TiledAccumulatorFSM.WRITING:
                with self._write_valid:
                    with self.write_row == self.array_size - 1:
                        self.write_state.next |= TiledAccumulatorFSM.IDLE
                        self.write_row.next |= 0
                    with otherwise:
                        self.write_addr.next |= self.write_addr + 1
                        self.write_row.next |= self.write_row + 1

    def _implement_read_fsm(self):
        read_base = self.base_addr_rom[self._read_tile_addr]

        self.read_addr_out <<= self.read_addr
        self.read_busy <<= self.read_state == ReadAccumulatorFSM.READING
        self.read_done <<= (self.read_state == ReadAccumulatorFSM.READING) & (
            self.read_row == self.array_size - 1
        )

        with conditional_assignment:
            with self.read_state == ReadAccumulatorFSM.IDLE:
                with self._read_start:
                    self.read_state.next |= ReadAccumulatorFSM.READING
                    self.read_addr.next |= read_base
                    self.read_row.next |= 0

            with self.read_state == ReadAccumulatorFSM.READING:
                with self.read_row == self.array_size - 1:
                    self.read_state.next |= ReadAccumulatorFSM.IDLE
                with otherwise:
                    self.read_addr.next |= self.read_addr + 1
                    self.read_row.next |= self.read_row + 1

    # Write interface methods
    def connect_tile_addr(self, addr: WireVector) -> None:
        self._tile_addr <<= addr

    def connect_write_start(self, start: WireVector) -> None:
        self._write_start <<= start

    def connect_write_mode(self, mode: WireVector) -> None:
        self._write_mode <<= mode

    def connect_write_valid(self, valid: WireVector) -> None:
        self._write_valid <<= valid

    # Read interface methods
    def connect_read_tile_addr(self, addr: WireVector) -> None:
        self._read_tile_addr <<= addr

    def connect_read_start(self, start: WireVector) -> None:
        self._read_start <<= start


class AccumulatorMemoryBank:
    """Integrated memory system for storing and accumulating systolic array outputs.

    Combines an address generator with parallel memory banks to provide a complete
    storage subsystem for matrix multiplication results. Supports both direct
    overwrite and accumulation modes, with independent read and write operations.

    Features:
    - Parallel memory banks (one per systolic array column)
    - Integrated address generation
    - Accumulate/overwrite modes
    - Independent read/write interfaces
    - Status signals for external coordination
    """

    def __init__(
        self,
        tile_addr_width: int,
        array_size: int,
        data_type: Type[BaseFloat],
        adder: Callable[[WireVector, WireVector, Type[BaseFloat]], WireVector],
    ):
        """Initialize the memory bank system.

        Args:
            tile_addr_width: Number of bits for addressing tiles. Determines
                number of tiles as 2^tile_addr_width.
            array_size: Number of parallel memory banks, matching systolic
                array dimension.
            data_type: Number format for stored data (e.g. BF16, Float8).
                Determines memory word width.
            adder: Function implementing addition for the specified data_type.
                Used for accumulation mode.
        """
        self.array_size = array_size
        self.tile_addr_width = tile_addr_width
        self.data_width = data_type.bitwidth()
        self.data_type = data_type
        self.adder = adder

        # Instantiate address generator
        self.addr_gen = TiledAddressGenerator(
            tile_addr_width=tile_addr_width, array_size=array_size
        )
        self.num_tiles = self.addr_gen.num_tiles

        # Input ports
        self._write_tile_addr = WireVector(self.tile_addr_width)
        self._write_start = WireVector(1)
        self._write_mode = WireVector(1)
        self._write_valid = WireVector(1)
        self._read_tile_addr = WireVector(self.tile_addr_width)
        self._read_start = WireVector(1)
        self._data_in = [WireVector(self.data_width) for i in range(array_size)]

        # Connect address generator
        self.addr_gen.connect_tile_addr(self._write_tile_addr)
        self.addr_gen.connect_write_start(self._write_start)
        self.addr_gen.connect_write_mode(self._write_mode)
        self.addr_gen.connect_write_valid(self._write_valid)
        self.addr_gen.connect_read_tile_addr(self._read_tile_addr)
        self.addr_gen.connect_read_start(self._read_start)

        # Create memory banks
        self.memory_banks = [
            MemBlock(
                bitwidth=self.data_width,
                addrwidth=self.addr_gen.internal_addr_width,
                name=f"bank_{i}",
            )
            for i in range(array_size)
        ]

        # Output ports
        self._data_out = [WireVector(self.data_width) for _ in range(array_size)]
        self.write_busy = self.addr_gen.write_busy
        self.write_done = self.addr_gen.write_done
        self.read_busy = self.addr_gen.read_busy
        self.read_done = self.addr_gen.read_done

        self._implement_memory_logic()

    def _implement_memory_logic(self):
        # Write logic
        for i, mem in enumerate(self.memory_banks):
            current_val = mem[self.addr_gen.write_addr_out]
            sum_result = self.adder(self._data_in[i], current_val, self.data_type)

            with conditional_assignment:
                with self.addr_gen.write_enable:
                    with self.addr_gen.write_mode_out:  # Accumulate mode
                        mem[self.addr_gen.write_addr_out] |= sum_result
                    with otherwise:  # Overwrite mode
                        mem[self.addr_gen.write_addr_out] |= self._data_in[i]

        # Read logic
        for i, mem in enumerate(self.memory_banks):
            with conditional_assignment:
                with self.read_busy:
                    self._data_out[i] |= mem[self.addr_gen.read_addr_out]

    def connect_inputs(
        self,
        write_tile_addr: WireVector | None = None,
        write_start: WireVector | None = None,
        write_mode: WireVector | None = None,
        write_valid: WireVector | None = None,
        read_tile_addr: WireVector | None = None,
        read_start: WireVector | None = None,
        data_in: list[WireVector] | None = None,
    ) -> None:
        """Connect all input control and data wires to the accumulator bank.

        Args:
            write_tile_addr: Address of tile to write to (tile_addr_width bits)
                Used to select which tile receives the incoming data.

            write_start: Start signal for write operation (1 bit)
                Pulses high for one cycle to initiate a new write sequence.

            write_mode: Mode selection for write operation (1 bit)
                0 = overwrite mode: new data replaces existing values
                1 = accumulate mode: new data is added to existing values

            write_valid: Data valid signal for write operation (1 bit)
                High when input data is valid and should be written/accumulated

            read_tile_addr: Address of tile to read from (tile_addr_width bits)
                Used to select which tile's data to output.

            read_start: Start signal for read operation (1 bit)
                Pulses high for one cycle to initiate a new read sequence.

            data_in: List of data input wires (data_width bits each)
                Input data from systolic array, one wire per column.
                Length must match array_size.

        Raises:
            AssertionError: If input wire widths don't match expected widths or
                        if data_in length doesn't match array_size.
        """
        if write_tile_addr is not None:
            assert len(write_tile_addr) == self.tile_addr_width
            self._write_tile_addr <<= write_tile_addr

        if write_start is not None:
            assert len(write_start) == 1
            self._write_start <<= write_start

        if write_mode is not None:
            assert len(write_mode) == 1
            self._write_mode <<= write_mode

        if write_valid is not None:
            assert len(write_valid) == 1
            self._write_valid <<= write_valid

        if read_tile_addr is not None:
            assert len(read_tile_addr) == self.tile_addr_width
            self._read_tile_addr <<= read_tile_addr

        if read_start is not None:
            assert len(read_start) == 1
            self._read_start <<= read_start

        if data_in is not None:
            assert (
                len(data_in) == self.array_size
            ), f"Expected {self.array_size} data inputs, got {len(data_in)}"
            for i, wire in enumerate(data_in):
                assert (
                    len(wire) == self.data_width
                ), f"Data input {i} width mismatch. Expected {self.data_width}, got {len(wire)}"
                self._data_in[i] <<= wire

    @property
    def write_interface(self) -> dict:
        return {
            "tile_addr": self._write_tile_addr,
            "start": self._write_start,
            "mode": self._write_mode,
            "valid": self._write_valid,
            "data": self._data_in,
        }

    @property
    def read_interface(self) -> dict:
        return {
            "tile_addr": self._read_tile_addr,
            "start": self._read_start,
            "data": self._data_out,
        }

    def get_output(self, bank: int) -> WireVector:
        return self._data_out[bank]
