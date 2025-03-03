from dataclasses import dataclass
from typing import Callable, Type, Dict
import numpy as np
import hashlib
import json

from pyrtl import (
    WireVector,
    Register,
    Output,
    Simulation,
    CompiledSimulation,
    concat,
)

from .adders import float_adder

from ..dtypes.bfloat16 import BF16


from .buffer import BufferMemory, WeightFIFO
from .systolic import SystolicArrayDiP
from .accumulators import Accumulator, TiledAccumulatorMemoryBank
from .activations import ReluState, ReluUnit
from ..dtypes import BaseFloat


@dataclass(unsafe_hash=True)
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

    @property
    def weight_tile_addr_width(self):
        """Get the width of the weight tile address bus in bits"""
        return (self.num_weight_tiles - 1).bit_length()


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
        self.outputs = [
            WireVector(config.accum_type.bitwidth()) for _ in range(config.array_size)
        ]

        # Connect components
        self._connect_components()

    def _create_control_wires(self):
        """Create unnamed WireVectors for control signals"""
        self.data_enable = WireVector(1)
        self.data_ins = [
            WireVector(self.config.data_type.bitwidth())
            for _ in range(self.config.array_size)
        ]

        self.weight_start_in = WireVector(1)
        self.weight_tile_addr_in = WireVector(self.fifo.tile_addr_width)

        self.accum_addr_in = WireVector(self.config.accum_addr_width)
        self.accum_mode_in = WireVector(1)

        self.act_start_in = WireVector(1)  # Whether to pass data to activation unit
        self.act_func_in = WireVector(1)  # Apply activation function or passthrough

    def _create_pipeline_registers(self):
        num_registers = self.config.array_size + 1 + int(self.config.pipeline)

        self.accum_addr_regs = [
            Register(self.config.accum_addr_width) for _ in range(num_registers)
        ]
        self.accum_addr_out = WireVector(self.config.accum_addr_width)
        self.accum_addr_out <<= self.accum_addr_regs[-1]

        self.accum_mode_regs = [Register(1) for _ in range(num_registers)]
        self.accum_mode_out = WireVector(1)
        self.accum_mode_out <<= self.accum_mode_regs[-1]

        self.act_start_regs = [Register(1) for _ in range(num_registers)]
        self.act_enable_regs = [Register(1) for _ in range(num_registers)]
        self.act_start_regs[0].next <<= self.act_start_in
        self.act_enable_regs[0].next <<= self.act_func_in

        # self.act_control_regs = [Register(2) for _ in range(num_registers)]
        # self.act_control_regs[0].next <<= concat(self.act_start_in, self.act_func_in)

        self.accum_addr_regs[0].next <<= self.accum_addr_in
        self.accum_mode_regs[0].next <<= self.accum_mode_in
        for i in range(1, len(self.accum_addr_regs)):
            self.accum_addr_regs[i].next <<= self.accum_addr_regs[i - 1]
            self.accum_mode_regs[i].next <<= self.accum_mode_regs[i - 1]
            # self.act_control_regs[i].next <<= self.act_control_regs[i - 1]
            if i < len(self.act_start_regs):
                self.act_enable_regs[i].next <<= self.act_enable_regs[i - 1]
                self.act_start_regs[i].next <<= self.act_start_regs[i - 1]

        self.act_addr = Register(self.config.accum_addr_width)
        self.act_func = Register(1)
        self.act_start = Register(1)

        self.act_addr.next <<= self.accum_addr_out
        # self.act_func.next <<= self.act_control_regs[-1][0]
        # self.act_start.next <<= self.act_control_regs[-1][1]
        self.act_func.next <<= self.act_enable_regs[-1]
        self.act_start.next <<= self.act_start_regs[-1]

    def _connect_components(self):
        """Internal component connections"""
        self._create_control_wires()
        self._create_pipeline_registers()

        # Connect buffer to external inputs
        self.fifo.connect_inputs(
            start=self.weight_start_in,
            tile_addr=self.weight_tile_addr_in,
        )

        self.systolic_array.connect_inputs(
            data_inputs=self.data_ins,
            enable_input=self.data_enable,
            weight_inputs=self.fifo.outputs.weights,
            weight_enable=self.fifo.outputs.active,
        )

        # Connect accumulator to systolic array
        self.accumulator.connect_inputs(
            data_in=self.systolic_array.results_out,
            write_addr=self.accum_addr_out,
            write_enable=self.systolic_array.control_out,
            write_mode=self.accum_mode_out,
            read_addr=self.act_addr,
            read_enable=self.act_start,
        )

        # Connect activation function to accumulator outputs
        self.activation.connect_inputs(
            inputs=self.accumulator.data_out,
            start=self.act_start,
            enable=self.act_func,
            valid=self.accumulator.read_enable,
        )
        self.activation.connect_outputs(self.outputs)

    def connect_inputs(
        self,
        data_enable: WireVector | None = None,
        data_inputs: list[WireVector] | None = None,
        weight_start: WireVector | None = None,
        weight_tile_addr: WireVector | None = None,
        accum_addr: WireVector | None = None,
        accum_mode: WireVector | None = None,
        act_start: WireVector | None = None,
        act_func: WireVector | None = None,
    ) -> None:
        """Connect input control wires to the accelerator.

        This method allows external control signals to be connected to the accelerator's
        internal control wires. All parameters are optional - only connected wires will
        be updated.

        Args:
            data_enable: 1-bit signal that enables data flow into the systolic array
            data_inputs: List of input data wires for the systolic array. Must match array_size
            weight_start: 1-bit signal that triggers loading of a new weight tile when pulsed high
            weight_tile_addr: Address selecting which weight tile to load from the FIFO.
                            Width must match the FIFO's tile address width
            accum_addr: Address for the accumulator memory bank. Width must match accum_addr_width
            accum_mode: 1-bit mode select (0=overwrite, 1=accumulate with existing values)
            act_start: 1-bit signal to enable passing data through the activation unit
            act_func: 1-bit signal to select activation function (0=passthrough, 1=ReLU)

        Raises:
            AssertionError: If input wire widths don't match expected widths or if data_inputs
                        length doesn't match array_size.
        """
        if data_enable is not None:
            assert len(data_enable) == 1, "Data enable signal must be 1 bit wide"
            self.data_enable <<= data_enable

        if data_inputs is not None:
            assert len(data_inputs) == self.config.array_size, (
                f"Number of data inputs must match array size. "
                f"Expected {self.config.array_size}, got {len(data_inputs)}"
            )
            for i, wire in enumerate(data_inputs):
                assert len(wire) == self.config.data_type.bitwidth(), (
                    f"Data input width mismatch. "
                    f"Expected {self.config.data_type.bitwidth()}, got {len(wire)}"
                )
                self.data_ins[i] <<= wire

        if weight_start is not None:
            assert len(weight_start) == 1, "Weight start signal must be 1 bit wide"
            self.weight_start_in <<= weight_start

        if weight_tile_addr is not None:
            assert len(weight_tile_addr) == self.fifo.tile_addr_width, (
                f"Weight tile address width mismatch. "
                f"Expected {self.fifo.tile_addr_width}, got {len(weight_tile_addr)}"
            )
            self.weight_tile_addr_in <<= weight_tile_addr

        if accum_addr is not None:
            assert len(accum_addr) == self.config.accum_addr_width, (
                f"Accumulator address width mismatch. "
                f"Expected {self.config.accum_addr_width}, got {len(accum_addr)}"
            )
            self.accum_addr_in <<= accum_addr

        if accum_mode is not None:
            assert len(accum_mode) == 1, "Accumulator mode must be 1 bit wide"
            self.accum_mode_in <<= accum_mode

        if act_start is not None:
            assert len(act_start) == 1, "Activation start signal must be 1 bit wide"
            self.act_start_in <<= act_start

        if act_func is not None:
            assert len(act_func) == 1, "Activation function select must be 1 bit wide"
            self.act_func_in <<= act_func

    def connect_outputs(
        self,
        outs: list[WireVector] | list[Output],
        valid: WireVector | Output | None = None,
    ):
        """Connect outputs to accelerator"""
        assert len(outs) == self.config.array_size, (
            f"Output width mismatch. "
            f"Expected {self.config.array_size}, got {len(outs)}"
        )
        for i, out in enumerate(outs):
            out <<= self.outputs[i]
        if valid is not None:
            assert len(valid) == 1, "Output valid signal must be a single bit wire"
            valid <<= self.activation.outputs_valid

    def inspect_systolic_array_state(self, sim: Simulation):
        """Return current PE array state"""
        return self.systolic_array.get_state(sim)

    def inspect_accumulator_state(self, sim: Simulation) -> np.ndarray:
        """Return all accumulator tiles as 3D array.

        Args:
            sim: PyRTL simulation instance

        Returns:
            2D numpy array of shape (2**accum_addr_width, array_size) containing
            all accumulator tile data converted to floating point values.
            Each tile contains array_size rows with array_size columns.
        """
        tiles = []
        for addr in range(2**self.config.accum_addr_width):
            row = [
                float(self.config.accum_type(binint=sim.inspect_mem(bank).get(addr, 0)))
                for bank in self.accumulator.memory_banks
            ]
            tiles.append(row)
        return np.array(tiles)

    def inspect_activation_state(self, sim: Simulation) -> ReluState:
        """Return current activation unit state"""
        return self.activation.inspect_state(sim)


@dataclass  # (frozen=True)
class CompiledAcceleratorConfig:
    """Configuration for a compiled accelerator."""

    array_size: int
    activation_type: Type[BaseFloat]
    weight_type: Type[BaseFloat]
    multiplier: Callable[[WireVector, WireVector, Type[BaseFloat]], WireVector]
    accum_addr_width: int = 12  # 4096 accumulator slots
    pipeline: bool = False

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Ensure activation dtype has bitwidth >= weight dtype
        if self.activation_type.bitwidth() < self.weight_type.bitwidth():
            raise ValueError(
                f"Activation dtype bitwidth ({self.activation_type.bitwidth()}) must be greater than or equal to "
                f"weight dtype bitwidth ({self.weight_type.bitwidth()})"
            )

    @property
    def name(self):
        dtype_name = lambda d: d.bitwidth() if d != BF16 else "b16"
        lmul = "-lmul" if "lmul" in self.multiplier.__name__.lower() else ""
        mem = f"-m{self.accum_addr_width}" if self.accum_addr_width != 12 else ""
        return (
            f"w{dtype_name(self.weight_type)}"
            f"a{dtype_name(self.activation_type)}"
            f"-{self.array_size}x{self.array_size}"
            f"{lmul}"
            f"{'-p' if self.pipeline else ''}"
            f"{mem}"
        )

    def __repr__(self) -> str:
        return (
            "CompiledAcceleratorConfig(\n"
            f"\tarray_size={self.array_size}\n"
            f"\tactivation_type={self.activation_type.__name__}\n"
            f"\tweight_type={self.weight_type.__name__}\n"
            f"\tmultiplier={self.multiplier.__name__}\n"
            f"\taccum_addr_width={self.accum_addr_width}\n"
            f"\tpipeline={self.pipeline}\n"
            # f'\tname="{self.name}"\n'
            ")"
        )

    def __hash__(self) -> int:
        """Generate a consistent hash value for this configuration.

        Returns:
            An integer hash value.
        """
        # Create a dictionary of the key configuration parameters
        config_dict = {
            "array_size": self.array_size,
            "activation_type": f"{self.activation_type.__module__}.{self.activation_type.__name__}",
            "weight_type": f"{self.weight_type.__module__}.{self.weight_type.__name__}",
            "multiplier": self.multiplier.__name__,
            "accum_addr_width": self.accum_addr_width,
            "pipeline": self.pipeline,
        }

        # Generate a hash from the sorted JSON representation
        hash_str = hashlib.sha256(
            json.dumps(config_dict, sort_keys=True).encode()
        ).hexdigest()

        # Convert the first 16 characters of the hex string to an integer
        return int(hash_str[:16], 16)

    @property
    def id(self):
        """Get a unique hexadecimal identifier for this configuration."""
        return hex(self.__hash__())[2:]


class CompiledAccelerator:
    def __init__(self, config: CompiledAcceleratorConfig):
        self.config = config

        # Instantiate hardware components
        self.systolic_array = SystolicArrayDiP(
            size=config.array_size,
            data_type=config.activation_type,
            weight_type=config.weight_type,
            accum_type=config.activation_type,
            multiplier=config.multiplier,
            adder=float_adder,
            pipeline=config.pipeline,
        )
        self.accumulator = Accumulator(
            addr_width=12,
            array_size=config.array_size,
            data_type=config.activation_type,
            adder=float_adder,
        )
        self.activation = ReluUnit(
            size=config.array_size,
            dtype=config.activation_type,
        )
        self.outputs = [
            WireVector(config.activation_type.bitwidth())
            for _ in range(config.array_size)
        ]

        # Connect components
        self._connect_components()

    def _create_control_wires(self):
        """Create unnamed WireVectors for control signals"""
        self.data_enable = WireVector(1)
        self.data_ins = [
            WireVector(self.config.activation_type.bitwidth())
            for _ in range(self.config.array_size)
        ]

        self.weight_enable = WireVector(1)
        self.weights_in = [
            WireVector(self.config.weight_type.bitwidth())
            for _ in range(self.config.array_size)
        ]

        self.accum_addr_in = WireVector(self.config.accum_addr_width)
        self.accum_mode_in = WireVector(1)

        self.act_start_in = WireVector(1)  # Whether to pass data to activation unit
        self.act_func_in = WireVector(1)  # Apply activation function or passthrough

    def _create_pipeline_registers(self):
        num_registers = self.config.array_size + 1 + int(self.config.pipeline)

        self.accum_addr_regs = [
            Register(self.config.accum_addr_width) for _ in range(num_registers)
        ]
        self.accum_addr_out = WireVector(self.config.accum_addr_width)
        self.accum_addr_out <<= self.accum_addr_regs[-1]

        self.accum_mode_regs = [Register(1) for _ in range(num_registers)]
        self.accum_mode_out = WireVector(1)
        self.accum_mode_out <<= self.accum_mode_regs[-1]

        self.act_control_regs = [Register(2) for _ in range(num_registers)]
        self.act_control_regs[0].next <<= concat(self.act_start_in, self.act_func_in)

        self.accum_addr_regs[0].next <<= self.accum_addr_in
        self.accum_mode_regs[0].next <<= self.accum_mode_in
        for i in range(1, len(self.accum_addr_regs)):
            self.accum_addr_regs[i].next <<= self.accum_addr_regs[i - 1]
            self.accum_mode_regs[i].next <<= self.accum_mode_regs[i - 1]
            self.act_control_regs[i].next <<= self.act_control_regs[i - 1]

        self.act_addr = Register(self.config.accum_addr_width)
        self.act_func = Register(1)
        self.act_start = Register(1)

        self.act_addr.next <<= self.accum_addr_out
        self.act_func.next <<= self.act_control_regs[-1][0]
        self.act_start.next <<= self.act_control_regs[-1][1]

    def _connect_components(self):
        """Internal component connections"""
        self._create_control_wires()
        self._create_pipeline_registers()

        # Connect buffer to external inputs
        self.systolic_array.connect_inputs(
            data_inputs=self.data_ins,
            enable_input=self.data_enable,
            weight_inputs=self.weights_in,
            weight_enable=self.weight_enable,
        )

        # Connect accumulator to systolic array
        self.accumulator.connect_inputs(
            data_in=self.systolic_array.results_out,
            write_addr=self.accum_addr_out,
            write_enable=self.systolic_array.control_out,
            write_mode=self.accum_mode_out,
            read_addr=self.act_addr,
            read_enable=self.act_start,
        )

        # Connect activation function to accumulator outputs
        self.activation.connect_inputs(
            inputs=self.accumulator.data_out,
            start=self.act_start,
            enable=self.act_func,
            valid=self.accumulator.read_enable,
        )
        self.activation.connect_outputs(self.outputs)

    def connect_inputs(
        self,
        data_enable: WireVector | None = None,
        data_inputs: list[WireVector] | None = None,
        weight_enable: WireVector | None = None,
        weights_in: list[WireVector] | None = None,
        accum_addr: WireVector | None = None,
        accum_mode: WireVector | None = None,
        act_start: WireVector | None = None,
        act_func: WireVector | None = None,
    ) -> None:
        """Connect input control wires to the accelerator.

        This method allows external control signals to be connected to the accelerator's
        internal control wires. All parameters are optional - only connected wires will
        be updated.

        Args:
            data_enable: 1-bit signal that enables data flow into the systolic array
            data_inputs: List of input data wires for the systolic array. Must match array_size
            weight_enable: 1-bit signal enable writing new weights to systolic array registers
            weights_in: List of input weight wires for the systolic array. Must match array_size
            accum_addr: Address for the accumulator memory bank. Width must match accum_addr_width
            accum_mode: 1-bit mode select (0=overwrite, 1=accumulate with existing values)
            act_start: 1-bit signal to enable passing data through the activation unit
            act_func: 1-bit signal to select activation function (0=passthrough, 1=ReLU)

        Raises:
            AssertionError: If input wire widths don't match expected widths or if data_inputs
                        length doesn't match array_size.
        """
        if data_enable is not None:
            assert len(data_enable) == 1, "Data enable signal must be 1 bit wide"
            self.data_enable <<= data_enable

        if data_inputs is not None:
            assert len(data_inputs) == self.config.array_size, (
                f"Number of data inputs must match array size. "
                f"Expected {self.config.array_size}, got {len(data_inputs)}"
            )
            for i, wire in enumerate(data_inputs):
                assert len(wire) == self.config.activation_type.bitwidth(), (
                    f"Data input width mismatch. "
                    f"Expected {self.config.activation_type.bitwidth()}, got {len(wire)}"
                )
                self.data_ins[i] <<= wire

        if weight_enable is not None:
            assert len(weight_enable) == 1, "Weight start signal must be 1 bit wide"
            self.weight_enable <<= weight_enable

        if weights_in is not None:
            assert len(weights_in) == self.config.array_size, (
                f"Weights input list length must match array size. "
                f"Expected {self.config.array_size}, got {len(weights_in)}"
            )
            for i, wire in enumerate(weights_in):
                assert len(wire) == self.config.weight_type.bitwidth(), (
                    f"Weight input wire width mismatch. "
                    f"Expected {self.config.weight_type.bitwidth()}, got {len(wire)}"
                )
                self.weights_in[i] <<= wire

        if accum_addr is not None:
            assert len(accum_addr) == self.config.accum_addr_width, (
                f"Accumulator address width mismatch. "
                f"Expected {self.config.accum_addr_width}, got {len(accum_addr)}"
            )
            self.accum_addr_in <<= accum_addr

        if accum_mode is not None:
            assert len(accum_mode) == 1, "Accumulator mode must be 1 bit wide"
            self.accum_mode_in <<= accum_mode

        if act_start is not None:
            assert len(act_start) == 1, "Activation start signal must be 1 bit wide"
            self.act_start_in <<= act_start

        if act_func is not None:
            assert len(act_func) == 1, "Activation function select must be 1 bit wide"
            self.act_func_in <<= act_func

    def connect_outputs(
        self,
        outs: list[WireVector] | list[Output],
        valid: WireVector | Output | None = None,
    ):
        """Connect outputs to accelerator"""
        assert len(outs) == self.config.array_size, (
            f"Output width mismatch. "
            f"Expected {self.config.array_size}, got {len(outs)}"
        )
        for i, out in enumerate(outs):
            out <<= self.outputs[i]
        if valid is not None:
            assert len(valid) == 1, "Output valid signal must be a single bit wire"
            valid <<= self.activation.outputs_valid

    def inspect_accumulator_state(self, sim: CompiledSimulation) -> np.ndarray:
        """Return all accumulator tiles as 3D array.

        Args:
            sim: PyRTL simulation instance

        Returns:
            2D numpy array of shape (2**accum_addr_width, array_size) containing
            all accumulator tile data converted to floating point values.
            Each tile contains array_size rows with array_size columns.
        """
        tiles = []
        for addr in range(2**self.config.accum_addr_width):
            row = [
                float(
                    self.config.activation_type(
                        binint=sim.inspect_mem(bank).get(addr, 0)
                    )
                )
                for bank in self.accumulator.memory_banks
            ]
            tiles.append(row)
        return np.array(tiles)


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
