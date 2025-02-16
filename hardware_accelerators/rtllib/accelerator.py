from dataclasses import dataclass
from os import read
from typing import Callable, Type, Dict, Any
import numpy as np
import pyrtl
from pyrtl import Input, Output, WireVector, Simulation, Register, MemBlock, RomBlock

from .buffer import BufferMemory, WeightFIFO
from .systolic import SystolicArrayDiP
from .accumulators import Accumulator, TiledAccumulatorMemoryBank
from .activations import ReluUnit
from ..dtypes import BaseFloat


@dataclass
class AcceleratorConfig:
    """Configuration class for a systolic array accelerator.

    This class defines the parameters and specifications for a systolic array
    accelerator including array dimensions, data types, arithmetic operations,
    and memory configuration.
    """

    array_size: int
    """Dimension of systolic array (always square)"""

    num_weight_tiles: int
    """Number of weight tiles in the FIFO. Each tile is equal to the size of the systolic array"""

    data_type: Type[BaseFloat]
    """Floating point format of input data to systolic array"""

    weight_type: Type[BaseFloat]
    """Floating point format of weight inputs"""

    accum_type: Type[BaseFloat]
    """Floating point format to accumulate values in"""

    pe_adder: Callable[[WireVector, WireVector, Type[BaseFloat]], WireVector]
    """Function to generate adder hardware for the processing elements"""

    accum_adder: Callable[[WireVector, WireVector, Type[BaseFloat]], WireVector]
    """Function to generate adder hardware for the accumulator buffer"""

    pe_multiplier: Callable[[WireVector, WireVector, Type[BaseFloat]], WireVector]
    """Function to generate multiplier hardware for the processing elements"""

    pipeline: bool
    """Whether to add a pipeline stage in processing elements between multiplier and adder"""

    accum_addr_width: int
    """Address width for accumulator memory. Determines number of individually addressable locations"""


class Accelerator:
    def __init__(self, config: AcceleratorConfig):
        self.config = config

        # Instantiate hardware components
        self.fifo = WeightFIFO(
            array_size=config.array_size,
            num_tiles=config.num_weight_tiles,
            dtype=config.weight_type,
        )
        self.systolic_array = SystolicArrayDiP(
            size=config.array_size,
            data_type=config.data_type,
            weight_type=config.weight_type,
            accum_type=config.accum_type,
            multiplier=config.pe_multiplier,
            adder=config.pe_adder,
            pipeline=config.pipeline,
            accum_addr_width=config.accum_addr_width,
        )
        self.accumulator = Accumulator(
            addr_width=config.accum_addr_width,
            array_size=config.array_size,
            data_type=config.accum_type,
            adder=config.accum_adder,
        )
        self.activation = ReluUnit(
            size=config.array_size,
            dtype=config.accum_type,
        )

        # Connect components
        self._connect_components()

    def _create_control_wires(self):
        """Create unnamed WireVectors for control signals"""
        self.data_ins = [
            WireVector(self.config.data_type.bitwidth())
            for _ in range(self.config.array_size)
        ]
        self.data_enable = WireVector(1)
        self.weight_start = WireVector(1)
        self.weight_tile_addr = WireVector(self.fifo.tile_addr_width)
        self.accum_addr = WireVector(self.config.accum_addr_width)
        self.accum_mode = WireVector(1)
        self.accum_read_start = WireVector(1)
        self.accum_read_addr = WireVector(self.config.accum_addr_width)
        self.enable_activation = WireVector(1)

    def _create_pipeline_registers(self):
        num_registers = self.config.array_size + 1 + int(self.config.pipeline)

        # self.accum_addr_in = WireVector(self.config.accum_addr_width)
        self.accum_addr_regs = [
            Register(self.config.accum_addr_width) for _ in range(num_registers)
        ]
        self.accum_addr_out = WireVector(self.config.accum_addr_width)
        self.accum_addr_out <<= self.accum_addr_regs[-1]

        # self.accum_mode_in = WireVector(1)
        self.accum_mode_regs = [Register(1) for _ in range(num_registers)]
        self.accum_mode_out = WireVector(1)
        self.accum_mode_out <<= self.accum_mode_regs[-1]

        self.accum_addr_regs[0].next <<= self.accum_addr
        self.accum_mode_regs[0].next <<= self.accum_mode
        for i in range(1, len(self.accum_addr_regs)):
            self.accum_addr_regs[i].next <<= self.accum_addr_regs[i - 1]
            self.accum_mode_regs[i].next <<= self.accum_mode_regs[i - 1]

    def _connect_components(self):
        """Internal component connections"""
        self._create_control_wires()
        self._create_pipeline_registers()

        # Connect buffer to external inputs
        self.fifo.connect_inputs(
            start=self.weight_start,
            tile_addr=self.weight_tile_addr,
        )

        self.systolic_array.connect_inputs(
            data_inputs=self.data_ins,
            enable_input=self.data_enable,
            weight_inputs=self.fifo.outputs.weights,
            weight_enable=self.fifo.outputs.active,
            accum_addr=self.accum_addr,
            accum_mode=self.accum_mode,
        )

        # Connect accumulator to systolic array
        self.accumulator.connect_inputs(
            data_in=self.systolic_array.results_out,
            write_addr=self.systolic_array.accum_addr_out,
            write_enable=self.systolic_array.control_out,
            write_mode=self.systolic_array.accum_mode_out,
            read_addr=self.accum_read_addr,
            read_enable=self.accum_read_start,
        )

        # Connect activation function to accumulator outputs
        self.activation.connect_inputs(
            inputs=self.accumulator.data_out,
            start=self.accum_read_start,
            enable=self.enable_activation,
            valid=self.accumulator.read_enable,
        )

    def step(
        self,
        data_vec: np.ndarray,
        accum_addr,
        accum_mode,
        activation,
        next_weight_tile,
    ):
        pass


@dataclass
class TiledAcceleratorConfig:
    """Configuration class for a systolic array accelerator.

    This class defines the parameters and specifications for a systolic array
    accelerator including array dimensions, data types, arithmetic operations,
    and memory configuration.
    """

    array_size: int
    """Dimension of systolic array (always square)"""

    data_type: Type[BaseFloat]
    """Floating point format of input data to systolic array"""

    weight_type: Type[BaseFloat]
    """Floating point format of weight inputs"""

    accum_type: Type[BaseFloat]
    """Floating point format to accumulate values in"""

    pe_adder: Callable[[WireVector, WireVector, Type[BaseFloat]], WireVector]
    """Function to generate adder hardware for the processing elements"""

    accum_adder: Callable[[WireVector, WireVector, Type[BaseFloat]], WireVector]
    """Function to generate adder hardware for the accumulator buffer"""

    pe_multiplier: Callable[[WireVector, WireVector, Type[BaseFloat]], WireVector]
    """Function to generate multiplier hardware for the processing elements"""

    pipeline: bool
    """Whether to add a pipeline stage in processing elements between multiplier and adder"""

    accumulator_tiles: int
    """Number of tiles in the accumulator memory, each tile is equal to the size of the systolic array"""

    @property
    def accum_addr_width(self):
        """Get the width of the accumulator address bus in bits"""
        return (self.accumulator_tiles - 1).bit_length() | 1


class TiledMatrixEngine:
    def __init__(self, config: TiledAcceleratorConfig):
        self.config = config

        # Create internal control wires (unnamed)
        self._create_control_wires()

        # Instantiate hardware components
        self.buffer = BufferMemory(
            config.array_size, config.data_type, config.weight_type
        )
        self.systolic_array = SystolicArrayDiP(
            size=config.array_size,
            data_type=config.data_type,
            weight_type=config.weight_type,
            accum_type=config.accum_type,
            multiplier=config.pe_multiplier,
            adder=config.pe_adder,
            pipeline=config.pipeline,
        )
        self.accumulator = TiledAccumulatorMemoryBank(
            tile_addr_width=config.accum_addr_width,
            array_size=config.array_size,
            data_type=config.accum_type,
            adder=config.accum_adder,
        )
        self.activation = ReluUnit(
            size=config.array_size,
            dtype=config.accum_type,
        )

        self.outputs = [
            WireVector(config.accum_type.bitwidth()) for _ in range(config.array_size)
        ]

        # Connect components
        self._connect_components()

    def _create_control_wires(self):
        """Create unnamed WireVectors for control signals"""
        self.data_start = WireVector(1)
        self.data_bank = WireVector(1)
        self.weight_start = WireVector(1)
        self.weight_bank = WireVector(1)
        self.accum_start = WireVector(1)
        self.accum_tile_addr = WireVector(self.config.accum_addr_width)
        self.accum_mode = WireVector(1)
        self.accum_read_start = WireVector(1)
        self.accum_read_tile_addr = WireVector(self.config.accum_addr_width)
        self.enable_activation = WireVector(1)

    def _connect_components(self):
        """Internal component connections"""
        # Connect buffer to external inputs
        self.buffer.connect_inputs(
            data_start=self.data_start,
            data_bank=self.data_bank,
            weight_start=self.weight_start,
            weight_bank=self.weight_bank,
        )

        # Connect systolic array to buffer
        buffer_outputs = self.buffer.get_outputs()
        self.systolic_array.connect_inputs(
            data_inputs=buffer_outputs.datas_out,
            weight_inputs=buffer_outputs.weights_out,
            enable_input=buffer_outputs.data_valid,
            weight_enable=buffer_outputs.weight_valid,
        )

        # Connect accumulator to systolic array
        self.accumulator.connect_inputs(
            data_in=self.systolic_array.results_out,
            write_start=self.accum_start,
            write_tile_addr=self.accum_tile_addr,
            write_mode=self.accum_mode,
            write_valid=self.systolic_array.control_out,
            read_start=self.accum_read_start,
            read_tile_addr=self.accum_read_tile_addr,
        )

        # Connect activation function to accumulator outputs
        self.activation.connect_inputs(
            inputs=self.accumulator.data_out,
            start=self.accum_read_start,
            enable=self.enable_activation,
            valid=self.accumulator.read_busy,
        )
        self.activation.connect_outputs(self.outputs)

    def connect_inputs(
        self,
        data_start: WireVector | None = None,
        data_bank: WireVector | None = None,
        weight_start: WireVector | None = None,
        weight_bank: WireVector | None = None,
        accum_start: WireVector | None = None,
        accum_tile_addr: WireVector | None = None,
        accum_mode: WireVector | None = None,
        accum_read_start: WireVector | None = None,
        accum_read_tile_addr: WireVector | None = None,
        enable_activation: WireVector | None = None,
    ) -> None:
        """Connect input control wires to the matrix engine.

        Args:
            data_start: 1-bit signal that triggers data streaming from buffer when pulsed high
            data_bank: 1-bit signal to select data memory bank (0 or 1)
            weight_start: 1-bit signal that triggers weight streaming from buffer when pulsed high
            weight_bank: 1-bit signal to select weight memory bank (0 or 1)
            accum_start: 1-bit signal that initiates accumulator write sequence when pulsed high
            accum_tile_addr: Address selecting which tile receives systolic array results. Latches on accum_start
            accum_mode: 1-bit mode select (0=overwrite, 1=accumulate with existing values)
            accum_read_start: 1-bit signal that initiates accumulator read sequence when pulsed high
            accum_read_tile_addr: Address selecting which tile's data to output
            enable_activation: 1-bit signal to enable activation function on output data
        Raises:
            AssertionError: If input wire widths don't match expected widths.
        """
        if data_start is not None:
            assert len(data_start) == 1, "Data start signal must be 1 bit wide"
            self.data_start <<= data_start

        if data_bank is not None:
            assert len(data_bank) == 1, "Data bank select must be 1 bit wide"
            self.data_bank <<= data_bank

        if weight_start is not None:
            assert len(weight_start) == 1, "Weight start signal must be 1 bit wide"
            self.weight_start <<= weight_start

        if weight_bank is not None:
            assert len(weight_bank) == 1, "Weight bank select must be 1 bit wide"
            self.weight_bank <<= weight_bank

        if accum_start is not None:
            assert len(accum_start) == 1, "Accumulator start signal must be 1 bit wide"
            self.accum_start <<= accum_start

        if accum_tile_addr is not None:
            assert (
                len(accum_tile_addr) == self.config.accum_addr_width
            ), f"Accumulator tile address width mismatch. Expected {self.config.accum_addr_width}, got {len(accum_tile_addr)}"
            self.accum_tile_addr <<= accum_tile_addr

        if accum_mode is not None:
            assert len(accum_mode) == 1, "Accumulator mode must be 1 bit wide"
            self.accum_mode <<= accum_mode

        if accum_read_start is not None:
            assert (
                len(accum_read_start) == 1
            ), "Accumulator read start signal must be 1 bit wide"
            self.accum_read_start <<= accum_read_start

        if accum_read_tile_addr is not None:
            assert (
                len(accum_read_tile_addr) == self.config.accum_addr_width
            ), f"Accumulator read tile address width mismatch. Expected {self.config.accum_addr_width}, got {len(accum_read_tile_addr)}"
            self.accum_read_tile_addr <<= accum_read_tile_addr

        if enable_activation is not None:
            assert (
                len(enable_activation) == 1
            ), "Enable activation signal must be 1 bit wide"
            self.enable_activation <<= enable_activation

    # Inspection methods
    def inspect_buffer_state(self, sim: Simulation) -> Dict[str, np.ndarray]:
        """Return current buffer contents as float arrays"""

        def read_mem(mem, dtype: Type[BaseFloat], reverse_rows=False):
            bitwidth = dtype.bitwidth()
            mask = (1 << bitwidth) - 1  # bitmask for one value
            rows = []
            for addr in range(self.config.array_size):
                packed_data = sim.inspect_mem(mem).get(addr, 0)
                row = []
                # Extract values from most-significant to least-significant bits:
                for i in range(self.config.array_size):
                    shift = (self.config.array_size - 1 - i) * bitwidth
                    extracted_bits = (packed_data >> shift) & mask
                    # Create a dtype instance from the binary integer:
                    row.append(dtype(binint=extracted_bits).decimal_approx)
                rows.insert(0, row)  # data is loaded backwards for systolic dataflow
            return np.array(rows)

        return {
            "data_banks": np.array(
                [
                    read_mem(self.buffer.data_mems[0], self.config.data_type),
                    read_mem(self.buffer.data_mems[1], self.config.data_type),
                ]
            ),
            "weight_banks": np.array(
                [
                    read_mem(self.buffer.weight_mems[0], self.config.weight_type),
                    read_mem(self.buffer.weight_mems[1], self.config.weight_type),
                ]
            ),
        }

    def inspect_systolic_array_state(self, sim: Simulation):
        """Return current PE array state"""
        return self.systolic_array.get_state(sim)

    def inspect_accumulator_state(self, sim: Simulation) -> np.ndarray:
        """Return all accumulator tiles as 3D array.

        Args:
            sim: PyRTL simulation instance

        Returns:
            3D numpy array of shape (num_tiles, array_size, array_size) containing
            all accumulator tile data converted to floating point values.
            Each tile contains array_size rows with array_size columns.
        """
        tiles = []
        for tile in range(self.config.accumulator_tiles):
            base_addr = tile * self.config.array_size
            tile_data = np.array(
                [
                    [
                        self.config.accum_type(
                            binint=sim.inspect_mem(bank).get(addr, 0)
                        ).decimal_approx
                        for bank in self.accumulator.memory_banks
                    ]
                    for addr in range(base_addr, base_addr + self.config.array_size)
                ]
            )
            tiles.append(tile_data)
        return np.array(tiles)

    def get_accumulator_outputs(self, sim: Simulation) -> np.ndarray:
        """Return current values on accumulator output ports"""
        return np.array(
            [
                self.config.accum_type(binint=sim.inspect(out.name)).decimal_approx
                for out in self.accumulator.read_interface["data"]
            ]
        )
