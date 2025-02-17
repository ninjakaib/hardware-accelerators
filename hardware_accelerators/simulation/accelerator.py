from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Self, Type
import numpy as np
import pyrtl
from pyrtl import (
    Input,
    Output,
    Simulation,
    Register,
    MemBlock,
    WireVector,
    reset_working_block,
)

from ..rtllib.multipliers import float_multiplier
from ..dtypes import BaseFloat, BF16
from ..rtllib.accumulators import TiledAccumulatorMemoryBank
from ..rtllib.adders import float_adder
from ..rtllib.systolic import SystolicArrayDiP, SystolicArraySimState
from ..rtllib.buffer import BufferMemory
from ..rtllib import (
    TiledAcceleratorConfig,
    TiledMatrixEngine,
    AcceleratorConfig,
    Accelerator,
)
from .matrix_utils import (
    pack_binary_vector,
    permutate_weight_matrix,
    convert_array_dtype,
)
from typing import TypedDict, Optional, Literal, NotRequired, Unpack


@dataclass
class TiledMatrixEngineSimState:
    step: int
    inputs: Dict[str, Any]
    buffer_state: Dict[str, np.ndarray]
    systolic_state: SystolicArraySimState
    accum_state: np.ndarray
    accum_outputs: np.ndarray

    def __repr__(self) -> str:
        """Pretty print the simulation state at this step"""
        width = 80
        sep = "=" * width
        subsep = "-" * width

        return (
            f"\n{sep}\n"
            f"Simulation Step {self.step}\n{sep}\n"
            f"Input Signals:\n{subsep}\n"
            f"  data_start: {self.inputs['data_start']}\n"
            f"  data_bank: {self.inputs['data_bank']}\n"
            f"  weight_start: {self.inputs['weight_start']}\n"
            f"  weight_bank: {self.inputs['weight_bank']}\n"
            f"  accum_start: {self.inputs['accum_start']}\n"
            f"  accum_tile_addr: {self.inputs['accum_tile_addr']}\n"
            f"  accum_mode: {self.inputs['accum_mode']}\n"
            f"  accum_read_start: {self.inputs['accum_read_start']}\n"
            f"  accum_read_tile_addr: {self.inputs['accum_read_tile_addr']}\n"
            f"\nBuffer Memory State:\n{subsep}\n"
            f"Data Bank 0:\n{np.array2string(self.buffer_state['data_banks'][0], precision=4, suppress_small=True)}\n"
            f"Data Bank 1:\n{np.array2string(self.buffer_state['data_banks'][1], precision=4, suppress_small=True)}\n"
            f"Weight Bank 0:\n{np.array2string(self.buffer_state['weight_banks'][0], precision=4, suppress_small=True)}\n"
            f"Weight Bank 1:\n{np.array2string(self.buffer_state['weight_banks'][1], precision=4, suppress_small=True)}\n"
            f"\nSystolic Array State:\n{subsep}\n{self.systolic_state}\n"
            # f"Weights Matrix:\n{np.array2string(self.systolic_state['weights'], precision=4, suppress_small=True)}\n"
            # f"Data Matrix:\n{np.array2string(self.systolic_state['data'], precision=4, suppress_small=True)}\n"
            # f"Accumulators:\n{np.array2string(self.systolic_state['accumulators'], precision=4, suppress_small=True)}\n"
            # f"Outputs:\n{np.array2string(self.systolic_state['outputs'], precision=4, suppress_small=True)}\n"
            # f"Controls:\n{self.systolic_state['controls']}\n"
            f"\nAccumulator Memory State:\n{subsep}\n"
            f"Tile States:\n"
            + "\n".join(
                f"Tile {i}:\n{np.array2string(tile, precision=4, suppress_small=True)}"
                for i, tile in enumerate(self.accum_state)
            )
            + f"\n\nAccumulator Output Ports:\n{np.array2string(self.accum_outputs, precision=4, suppress_small=True)}\n"
            f"{sep}\n"
        )


class TiledMatrixEngineSimulator:
    def __init__(self, config: TiledAcceleratorConfig):
        self.config = config
        self.history: List[TiledMatrixEngineSimState] = []
        self._setup()

    def _setup(self):
        reset_working_block()
        self.engine = TiledMatrixEngine(self.config)
        self._create_sim_inputs()
        self.sim = Simulation()

    def _create_sim_inputs(self):
        """Create named Inputs and connect to engine"""
        inputs = {
            "data_start": Input(1, "data_start"),
            "data_bank": Input(1, "data_bank"),
            "weight_start": Input(1, "weight_start"),
            "weight_bank": Input(1, "weight_bank"),
            "accum_start": Input(1, "accum_start"),
            "accum_tile_addr": Input(self.config.accum_addr_width, "accum_tile_addr"),
            "accum_mode": Input(1, "accum_mode"),
            "accum_read_start": Input(1, "accum_read_start"),
            "accum_read_tile_addr": Input(
                self.config.accum_addr_width, "accum_read_tile_addr"
            ),
            "enable_activation": Input(1, "enable_activation"),
        }
        self.engine.connect_inputs(**inputs)
        return inputs

    def _get_default_inputs(self) -> Dict[str, int]:
        """Generate default input values"""
        return {
            "data_start": 0,
            "data_bank": 0,
            "weight_start": 0,
            "weight_bank": 0,
            "accum_start": 0,
            "accum_tile_addr": 0,
            "accum_mode": 0,
            "accum_read_start": 0,
            "accum_read_tile_addr": 0,
            "enable_activation": 0,
        }

    def load_weights(self, weights: np.ndarray, bank: int):
        """Directly load weights into buffer memory (pre-permutated)"""
        if weights.shape != (self.config.array_size, self.config.array_size):
            raise ValueError(
                f"Weights must be {self.config.array_size}x{self.config.array_size}"
            )

        weights = permutate_weight_matrix(weights)
        weight_mem = self.sim.inspect_mem(self.engine.buffer.weight_mems[bank])

        # Load directly into memory bank
        for i, row in enumerate(weights[::-1]):
            weight_mem[i] = self._vec_to_binary(row, self.config.weight_type)

    def load_activations(self, activations: np.ndarray, bank: int):
        """Directly load activations into buffer memory"""
        if activations.shape != (self.config.array_size, self.config.array_size):
            raise ValueError(
                f"Data must be {self.config.array_size}x{self.config.array_size}"
            )

        data_mem = self.sim.inspect_mem(self.engine.buffer.data_mems[bank])

        # Load directly into memory bank
        for i, row in enumerate(activations[::-1]):
            data_mem[i] = self._vec_to_binary(row, self.config.data_type)

    def _vec_to_binary(self, vec, dtype: Type[BaseFloat]):
        """Convert vector to concatenated binary representation"""
        concatenated = 0
        for i, d in enumerate(vec[::-1]):
            binary = dtype(d).binint
            concatenated += binary << (i * dtype.bitwidth())
        return concatenated

    # TODO: change to control the activation module
    def read_accumulator_tile(self, tile: int) -> np.ndarray:
        """Read tile from accumulator memory"""

        # Capture outputs over array_size cycles
        results = []
        self.step(accum_read_tile_addr=tile, accum_read_start=1)
        for _ in range(self.config.array_size):
            self.step()
            results.append(self.engine.get_accumulator_outputs(self.sim))

        return np.array(results)  # [-self.config.array_size :]

    def matmul(
        self, data_bank: int, weight_bank: int, accum_tile: int, accumulate: bool
    ):
        """Perform matrix multiply with current buffers and store result"""
        # Start streaming weights into systolic array
        self.step(weight_bank=weight_bank, weight_start=1)

        # Load all except the "last" (actually 1st) row of weights
        for _ in range(self.config.array_size - 2):
            self.step()

        # Start streaming data into systolic array
        # actual data stream is delayed by 1 cycle so we have to start when there are 2 rows of weights left
        self.step(data_bank=data_bank, data_start=1)

        # Stream activations until systolic accumulators are populated with results
        # for _ in range(2 * self.config.array_size + self.config.pipeline - 2):
        for _ in range(self.config.array_size + self.config.pipeline):
            self.step()

        # Prepare accumulator to accept results
        self.step(accum_tile_addr=accum_tile, accum_mode=int(accumulate), accum_start=1)

        # Populate the accumulator memory as results flow out of the systolic array
        for _ in range(self.config.array_size):
            self.step()

    def step(self, **kwargs):
        inputs = self._get_default_inputs()
        inputs.update(kwargs)
        self.sim.step(inputs)

        self.history.append(
            TiledMatrixEngineSimState(
                step=len(self.history),
                inputs=inputs,
                buffer_state=self.engine.inspect_buffer_state(self.sim),
                systolic_state=self.engine.inspect_systolic_array_state(self.sim),
                accum_state=self.engine.inspect_accumulator_state(self.sim),
                accum_outputs=self.engine.get_accumulator_outputs(self.sim),
            )
        )

    @classmethod
    def calculate_matmul(cls, weights: np.ndarray, data: np.ndarray):
        # Ensure weights and data are square matrices of the same size
        assert weights.shape[0] == weights.shape[1], "Weights must be a square matrix"
        assert data.shape[0] == data.shape[1], "Data must be a square matrix"
        assert (
            weights.shape[0] == data.shape[0]
        ), "Weights and data must be the same size"
        config = TiledAcceleratorConfig(
            array_size=weights.shape[0],
            data_type=BF16,
            weight_type=BF16,
            accum_type=BF16,
            pe_adder=float_adder,
            accum_adder=float_adder,
            pe_multiplier=float_multiplier,
            pipeline=False,
            accumulator_tiles=1,
        )
        sim = cls(config)
        sim.load_weights(weights, bank=0)
        sim.load_activations(data, bank=0)
        sim.matmul(0, 0, 0, False)
        return sim.read_accumulator_tile(0)[::-1]

    @property
    def data_banks(self) -> np.ndarray:
        """Get current state of data memory banks.

        Returns:
            List of numpy arrays containing the current values in each data bank.
            Index 0 is bank 0, index 1 is bank 1.
        """
        buffer_state = self.engine.inspect_buffer_state(self.sim)
        return buffer_state["data_banks"]

    @property
    def weight_banks(self) -> np.ndarray:
        """Get current state of weight memory banks.

        Returns:
            List of numpy arrays containing the current values in each weight bank.
            Index 0 is bank 0, index 1 is bank 1.
        """
        buffer_state = self.engine.inspect_buffer_state(self.sim)
        return buffer_state["weight_banks"]

    @property
    def systolic_weights(self) -> np.ndarray:
        """Get current weight values held in the systolic array PEs.

        Returns:
            2D numpy array containing the weight values currently loaded in each PE.
            Shape is (array_size, array_size).
        """
        systolic_state = self.engine.inspect_systolic_array_state(self.sim)
        return systolic_state.weights

    @property
    def systolic_data(self) -> np.ndarray:
        """Get current data values held in the systolic array PEs.

        Returns:
            2D numpy array containing the data values currently loaded in each PE.
            Shape is (array_size, array_size).
        """
        systolic_state = self.engine.inspect_systolic_array_state(self.sim)
        return systolic_state.data

    @property
    def systolic_accumulators(self) -> np.ndarray:
        """Get current accumulator values in the systolic array PEs.

        Returns:
            2D numpy array containing the accumulator values in each PE.
            Shape is (array_size, array_size).
        """
        systolic_state = self.engine.inspect_systolic_array_state(self.sim)
        return systolic_state.accumulators

    @property
    def systolic_outputs(self) -> np.ndarray:
        """Get current output values from the systolic array PEs.

        Returns:
            2D numpy array containing the current output values from each PE.
            Shape is (array_size, array_size).
        """
        systolic_state = self.engine.inspect_systolic_array_state(self.sim)
        return systolic_state.outputs

    @property
    def accumulator_memory(self) -> np.ndarray:
        """Get current state of all accumulator memory tiles.

        Returns:
            3D numpy array containing all accumulator tile contents.
            Shape is (num_tiles, array_size, array_size).
        """
        return self.engine.inspect_accumulator_state(self.sim)

    @property
    def accumulator_outputs(self) -> np.ndarray:
        """Get current values on accumulator output ports.

        Returns:
            1D numpy array containing the current values on accumulator output ports.
            Length is array_size.
        """
        return self.engine.get_accumulator_outputs(self.sim)

    # Then update print_state to use these properties
    def print_history(
        self,
        inputs: bool = True,
        memory_buffer: bool = True,
        systolic_array: bool = True,
        accumulator: bool = True,
    ):
        """Print selected components from simulation history.

        Args:
            inputs: Whether to print input signals
            memory_buffer: Whether to print buffer memory state
            systolic_array: Whether to print systolic array state
            accumulator: Whether to print accumulator state
        """
        width = 80
        sep = "=" * width
        subsep = "-" * width

        for state in self.history:
            print(f"\n{sep}\nSimulation Step {state.step}\n{sep}")

            if inputs:
                print(f"\nInput Signals:\n{subsep}")
                print(f"  data_start: {state.inputs['data_start']}")
                print(f"  data_bank: {state.inputs['data_bank']}")
                print(f"  weight_start: {state.inputs['weight_start']}")
                print(f"  weight_bank: {state.inputs['weight_bank']}")
                print(f"  accum_start: {state.inputs['accum_start']}")
                print(f"  accum_tile_addr: {state.inputs['accum_tile_addr']}")
                print(f"  accum_mode: {state.inputs['accum_mode']}")
                print(f"  accum_read_start: {state.inputs['accum_read_start']}")
                print(f"  accum_read_tile_addr: {state.inputs['accum_read_tile_addr']}")

            if memory_buffer:
                print(f"\nBuffer Memory State:\n{subsep}")
                print(
                    f"Data Bank 0:\n{np.array2string(state.buffer_state['data_banks'][0], precision=4, suppress_small=True)}"
                )
                print(
                    f"Data Bank 1:\n{np.array2string(state.buffer_state['data_banks'][1], precision=4, suppress_small=True)}"
                )
                print(
                    f"Weight Bank 0:\n{np.array2string(state.buffer_state['weight_banks'][0], precision=4, suppress_small=True)}"
                )
                print(
                    f"Weight Bank 1:\n{np.array2string(state.buffer_state['weight_banks'][1], precision=4, suppress_small=True)}"
                )

            if systolic_array:
                print(f"\nSystolic Array State:\n{subsep}\n")
                print(state.systolic_state)

            if accumulator:
                print(f"\nAccumulator Memory State:\n{subsep}")
                print("Tile States:")
                for i, tile in enumerate(state.accum_state):
                    print(
                        f"Tile {i}:\n{np.array2string(tile, precision=4, suppress_small=True)}"
                    )
                print(
                    f"\nAccumulator Output Ports:\n{np.array2string(state.accum_outputs, precision=4, suppress_small=True)}"
                )

            print("\n\n")


@dataclass
class AcceleratorSimState:
    step: int
    inputs: Dict[str, Any]
    systolic_state: SystolicArraySimState
    # fifo_state: Dict[str, np.ndarray]
    accum_state: np.ndarray
    outputs: np.ndarray

    def __repr__(self) -> str:
        """Pretty print the simulation state at this step"""
        width = 80
        sep = "=" * width
        subsep = "-" * width

        return (
            f"\n{sep}\n"
            f"Simulation Step {self.step}\n{sep}\n"
            f"Input Signals:\n{subsep}\n"
            f"  data_enable: {self.inputs['data_enable']}\n"
            f"  data_inputs: {self.inputs['data_inputs']}\n"
            f"  weight_start: {self.inputs['weight_start']}\n"
            f"  weight_tile_addr: {self.inputs['weight_tile_addr']}\n"
            f"  accum_addr: {self.inputs['accum_addr']}\n"
            f"  accum_mode: {self.inputs['accum_mode']}\n"
            f"  act_start: {self.inputs['act_start']}\n"
            f"  act_func: {self.inputs['act_func']}\n"
            f"\nSystolic Array State:\n{subsep}\n{self.systolic_state}\n"
            # f"\nFIFO State:\n{subsep}\n"
            # f"  Weight FIFO:\n{np.array2string(self.fifo_state['weight_fifo'], precision=4, suppress_small=True)}\n"
            # f"  Data FIFO:\n{np.array2string(self.fifo_state['data_fifo'], precision=4, suppress_small=True)}\n"
            f"\nAccumulator State:\n{subsep}\n"
            f"{np.array2string(self.accum_state, precision=4, suppress_small=True)}\n"
            f"\nOutputs:\n{np.array2string(self.outputs, precision=4, suppress_small=True)}\n"
            f"{sep}\n"
        )


# New TypedDict for step parameters
class StepInputs(TypedDict):
    """Type hints for simulator step inputs with documentation for each control signal.

    All fields are optional (NotRequired) and default to 0 if not specified.
    """

    data_enable: NotRequired[int]
    """1-bit signal that enables data flow into the systolic array"""

    weight_start: NotRequired[int]
    """1-bit signal that triggers loading of a new weight tile when pulsed high"""

    weight_tile_addr: NotRequired[int]
    """Address selecting which weight tile to load from the FIFO"""

    accum_addr: NotRequired[int]
    """Address for the accumulator memory bank"""

    accum_mode: NotRequired[int]
    """1-bit mode select (0=overwrite, 1=accumulate with existing values)"""

    act_start: NotRequired[int]
    """1-bit signal to enable passing data through the activation unit"""

    act_func: NotRequired[int]
    """1-bit signal to select activation function (0=passthrough, 1=ReLU)"""

    data: NotRequired[np.ndarray]
    """Input data vector of length array_size. Will set data_enable=1 if provided"""


class AcceleratorSimulator:
    """Simulator for the Accelerator hardware design.

    This simulator provides a simplified interface compared to the TiledMatrixEngine,
    focusing on step-by-step execution with more straightforward input handling.
    """

    def __init__(self, config: AcceleratorConfig):
        self.config = config
        self.history: List[AcceleratorSimState] = []
        self.setup()

    @classmethod
    def default_config(cls, array_size: int, num_weight_tiles: int) -> Self:
        """Create a default configuration for the accelerator"""
        return cls(
            AcceleratorConfig(
                array_size=array_size,
                num_weight_tiles=num_weight_tiles,
                data_type=BF16,
                weight_type=BF16,
                accum_type=BF16,
                pe_adder=float_adder,
                accum_adder=float_adder,
                pe_multiplier=float_multiplier,
                pipeline=False,
                accum_addr_width=array_size,
            )
        )

    def setup(self):
        """Initialize the simulation environment and hardware"""
        if not hasattr(self, "accelerator"):
            reset_working_block()
            self.accelerator = Accelerator(self.config)

            # Create simulation inputs with clear names
            self.inputs = {
                "data_enable": Input(1, "data_enable"),
                "data_inputs": [
                    Input(self.config.data_type.bitwidth(), f"data_in_{i}")
                    for i in range(self.config.array_size)
                ],
                "weight_start": Input(1, "weight_start"),
                "weight_tile_addr": Input(
                    self.config.weight_tile_addr_width, "weight_tile_addr"
                ),
                "accum_addr": Input(self.config.accum_addr_width, "accum_addr"),
                "accum_mode": Input(1, "accum_mode"),
                "act_start": Input(1, "act_start"),
                "act_func": Input(1, "act_func"),
            }

            # Connect inputs to accelerator
            self.accelerator.connect_inputs(**self.inputs)

        self.sim = Simulation()
        self.history = []

    def execute_instruction(
        self,
        data_vec: np.ndarray | None = None,
        accum_addr: int = 0,
        accum_mode: int = 0,  # 0=overwrite, 1=accumulate
        activation_enable: bool = False,  # 0=nothing, 1=enable
        activation_func: Literal["passthrough", "relu"] = "passthrough",
        load_new_weights: bool = False,  # 0=don't load, 1=load (blocking)
        weight_tile_addr: int = 0,  # address of weight tile to load
        flush_pipeline: bool = True,
        nop: bool = False,
    ) -> None:
        """Execute one instruction with the given inputs. Will step more than one cycle

        Args:
            data_vec: Input data vector of length array_size. If None, data_enable will be 0
            accum_addr: Address in accumulator memory to read/write
            accum_mode: 0 for overwrite, 1 for accumulate with existing values
            activation_enable: Whether to enable the activation unit
            activation_func: Activation function to use (passthrough or ReLU)
            load_new_weights: Whether to load a new weight tile from FIFO memory
            weight_tile_addr: Address of weight tile to load when weights=1
            flush_pipeline: Whether to flush the systolic array after input data to allow for new data/weights on the next cycle. If False, the systolic array will be held in the same state for the next cycle allowing for multiple data inputs to be streamed in.
            nop: No operation, step 1 cycle with no inputs
        """
        if nop:
            self.step()
            return

        if load_new_weights:
            # Start streaming weights into systolic array
            self.step(weight_start=1, weight_tile_addr=weight_tile_addr)

            # Load all except the "last" (actually 1st) row of weights
            for _ in range(self.config.array_size - 1):
                self.step()

        if data_vec is not None:
            # Start streaming data into systolic array at the same time as last weights
            self.step(
                data=data_vec,
                accum_addr=accum_addr,
                accum_mode=accum_mode,
                act_start=activation_enable,
                act_func=1 if activation_func == "relu" else 0,
            )

            if flush_pipeline:
                for _ in range(self.config.array_size + self.config.pipeline):
                    self.step()
        else:
            self.step()

    def step(
        self,
        **kwargs: Unpack[StepInputs],
    ) -> None:
        """Execute a simulation step with the given inputs.

        Args:
            data: Input data vector of length array_size. If provided, data_enable will be set to 1
            **kwargs: Additional control signals
        """
        data = kwargs.pop("data", None)
        inputs = self.get_sim_inputs(**kwargs)

        # Handle data input array
        if data is not None:
            if len(data) != self.config.array_size:
                raise ValueError(f"Data length must be {self.config.array_size}")
            inputs["data_enable"] = 1
            for i, value in enumerate(data):
                inputs[f"data_in_{i}"] = self.config.data_type(value).binint

        self.sim.step(inputs)
        inputs.update(data_inputs=data)  # type: ignore
        self._update_history(inputs)

    def _update_history(self, inputs):
        self.history.append(
            AcceleratorSimState(
                step=len(self.history),
                inputs=inputs,
                systolic_state=self.accelerator.inspect_systolic_array_state(self.sim),
                accum_state=self.accelerator.inspect_accumulator_state(self.sim),
                outputs=self._get_outputs(),
            )
        )

    def get_sim_inputs(self, **kwargs) -> Dict[str, int]:
        """Get default input values"""
        defaults = {
            "data_enable": 0,
            "weight_start": 0,
            "weight_tile_addr": 0,
            "accum_addr": 0,
            "accum_mode": 0,
            "act_start": 0,
            "act_func": 0,
            **{f"data_in_{i}": 0 for i in range(self.config.array_size)},
        }
        for key, value in kwargs.items():
            if key in defaults:
                defaults[key] = value
            else:
                Warning(f"Invalid input key: {key}")
        return defaults

    def _get_outputs(self) -> np.ndarray:
        """Get current output values converted to floating point"""
        return np.array(
            [
                self.config.accum_type(binint=self.sim.inspect(out.name)).decimal_approx
                for out in self.accelerator.outputs
            ]
        )

    def load_weights(self, weights: np.ndarray, tile_addr: int, permutate: bool = True):
        """Load a weight matrix into the specified FIFO tile.

        Args:
            weights: Array of shape (array_size, array_size) containing weight values
            tile_addr: Address of tile to load weights into
            permutate: (default True) Whether to arrange the weights for the DiP systolic array
        """
        if weights.shape != (self.config.array_size, self.config.array_size):
            raise ValueError(
                f"Weights must be {self.config.array_size}x{self.config.array_size}"
            )

        # Convert weights to binary and load into FIFO memory
        weight_mem = self.sim.inspect_mem(self.accelerator.fifo.memory)
        base_addr = tile_addr * self.config.array_size

        if permutate:
            weights = permutate_weight_matrix(weights)

        for i, row in enumerate(weights[::-1]):
            weight_mem[base_addr + i] = pack_binary_vector(row, self.config.weight_type)

    def print_state(self, step: int | None = None) -> None:
        """Print the simulation state at the specified step.

        Args:
            step: Step number to print. If None, prints the latest step.
        """
        if step is None:
            state = self.history[-1]
        else:
            state = self.history[step]

        print(f"\n{'='*80}\nSimulation Step {state.step}\n{'='*80}")

        # Print inputs
        print("\nInputs:")
        print(f"  Data Enable: {state.inputs['data_enable']}")
        if state.inputs["data_enable"]:
            data_values = [
                state.inputs[f"data_in_{i}"] for i in range(self.config.array_size)
            ]
            print(f"  Data Values: {data_values}")
        print(f"  Weight Start: {state.inputs['weight_start']}")
        print(f"  Weight Tile Addr: {state.inputs['weight_tile_addr']}")
        print(f"  Accumulator Addr: {state.inputs['accum_addr']}")
        print(f"  Accumulator Mode: {state.inputs['accum_mode']}")
        print(f"  Activation Start: {state.inputs['act_start']}")
        print(f"  Activation Function: {state.inputs['act_func']}")

        # Print hardware state
        print("\nSystolic Array State:")
        print(state.systolic_state)

        # print("\nFIFO State:")
        # print(state.fifo_state)

        print("\nAccumulator State:")
        print(np.array2string(state.accum_state, precision=4, suppress_small=True))

        print("\nOutputs:")
        print(np.array2string(state.outputs, precision=4, suppress_small=True))
