from pyrtl import WireVector, Register, RomBlock, conditional_assignment, otherwise
from enum import IntEnum


class TiledAccumulatorFSM(IntEnum):
    """FSM States for Accumulator Control

    State Transitions:
    IDLE -> WRITING: When start signal received
    WRITING -> WRITING: While processing rows within tile
    WRITING -> IDLE: Immediate transition when last row of tile processed
    """

    IDLE = 0
    """IDLE: Waiting for new tile operation"""
    WRITING = 1
    """WRITING: Processing rows of systolic array output"""


class TiledAddressGenerator:
    """Generates addresses and control signals for accumulator bank memory

    This module manages the storage of systolic array outputs into tile-organized
    memory. It automatically handles address generation and increments within tiles,
    abstracting away the internal memory organization from the higher-level control.
    """

    def __init__(self, tile_addr_width: int, array_size: int):
        """Initialize address generator

        Args:
            tile_addr_width: Number of bits for addressing tiles
            array_size: Dimension of systolic array (NxN)
        """
        # Configuration parameters
        self.array_size = array_size
        self.num_tiles = 2**tile_addr_width

        # Calculate required address width based on total storage needed
        # (num_tiles * rows_per_tile)
        self.internal_addr_width = (self.num_tiles * array_size - 1).bit_length()

        # Create base address lookup ROM
        # For example, with 4x4 array and 4 tiles:
        # tile 0 -> base addr 0
        # tile 1 -> base addr 4
        # tile 2 -> base addr 8
        # tile 3 -> base addr 12
        base_addrs = [i * array_size for i in range(self.num_tiles)]
        self.base_addr_rom = RomBlock(
            bitwidth=self.internal_addr_width,
            addrwidth=tile_addr_width,
            romdata=base_addrs,
        )

        # Input signals
        self._tile_addr = WireVector(tile_addr_width)
        self._start = WireVector(1)
        self._write_valid = WireVector(1)

        # State registers
        self.state = Register(1)  # Current FSM state
        self.internal_addr = Register(self.internal_addr_width)
        self.current_row = Register(array_size.bit_length())

        # Output signals
        self.internal_write_addr = WireVector(self.internal_addr_width)
        self.internal_write_enable = WireVector(1)
        self.busy = WireVector(1)
        self.tile_complete = WireVector(1)

        # Implement FSM logic
        self._implement_fsm()

    def _implement_fsm(self):
        """Implements the FSM logic using conditional assignments"""
        # Get base address from ROM using tile address
        tile_base = self.base_addr_rom[self._tile_addr]

        # Set output signals (combinational, outside conditional)
        self.internal_write_addr <<= self.internal_addr
        self.internal_write_enable <<= (
            self.state == TiledAccumulatorFSM.WRITING
        ) & self._write_valid
        self.busy <<= self.state == TiledAccumulatorFSM.WRITING
        self.tile_complete <<= (
            (self.state == TiledAccumulatorFSM.WRITING)
            & self._write_valid
            & (self.current_row == self.array_size)
        )

        # FSM Logic
        with conditional_assignment:
            # IDLE: Wait for start signal
            with self.state == TiledAccumulatorFSM.IDLE:
                with self._start:
                    self.state.next |= TiledAccumulatorFSM.WRITING
                    self.internal_addr.next |= tile_base
                    self.current_row.next |= 0

            # WRITING: Process rows until tile complete
            with self.state == TiledAccumulatorFSM.WRITING:
                with self._write_valid:
                    # Only return to IDLE after last row is processed
                    with self.current_row == self.array_size - 1:
                        self.state.next |= TiledAccumulatorFSM.IDLE
                        self.current_row.next |= 0
                    with otherwise:
                        self.internal_addr.next |= self.internal_addr + 1
                        self.current_row.next |= self.current_row + 1

    # Connection methods
    def connect_tile_addr(self, addr: WireVector) -> None:
        """Connect tile address input

        Args:
            addr: Address of tile to write to/read from (tile_addr_width bits)
        """
        self._tile_addr <<= addr

    def connect_start(self, start: WireVector) -> None:
        """Connect start signal

        Args:
            start: Signal to begin processing new tile (1 bit)
        """
        self._start <<= start

    def connect_write_valid(self, valid: WireVector) -> None:
        """Connect write valid signal from systolic array

        Args:
            valid: Signal indicating valid output from systolic array (1 bit)
        """
        self._write_valid <<= valid

    def get_write_addr(self) -> WireVector:
        """Get current write address output

        Returns:
            WireVector of internal_addr_width bits
        """
        return self.internal_write_addr

    def get_write_enable(self) -> WireVector:
        """Get write enable output

        Returns:
            1-bit WireVector, high when writing should occur
        """
        return self.internal_write_enable

    def get_busy(self) -> WireVector:
        """Get busy status

        Returns:
            1-bit WireVector, high when processing a tile
        """
        return self.busy

    def get_tile_complete(self) -> WireVector:
        """Get tile completion signal

        Returns:
            1-bit WireVector, pulses high when tile is complete
        """
        return self.tile_complete

    def get_base_addr(self, tile_addr: int) -> int:
        """Get base address for a given tile

        Args:
            tile_addr: Tile address to lookup

        Returns:
            Base address for the specified tile
        """
        return tile_addr * self.array_size

    @property
    def output_ports(self):
        """Get dictionary of output ports"""
        return {
            "write_addr": self.internal_write_addr,
            "write_enable": self.internal_write_enable,
            "busy": self.busy,
            "tile_complete": self.tile_complete,
        }

    @property
    def input_ports(self):
        """Get dictionary of input ports"""
        return {
            "tile_addr": self._tile_addr,
            "start": self._start,
            "write_valid": self._write_valid,
        }
