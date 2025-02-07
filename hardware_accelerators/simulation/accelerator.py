from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Type
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
from ..dtypes import BaseFloat, BF16
from ..rtllib.accumulators import AccumulatorMemoryBank
from ..rtllib.adders import float_adder
from ..rtllib.systolic import SystolicArrayDiP
from ..rtllib.buffer import BufferMemory
from ..rtllib import AcceleratorConfig, MatrixEngine
from .utils import permutate_weight_matrix, convert_array_dtype


# @dataclass
# class SimulationState:
#     """Complete system state at a simulation step"""
#     step: int
#     buffer_data: np.ndarray
#     buffer_weights: np.ndarray
#     systolic_weights: np.ndarray
#     systolic_data: np.ndarray
#     accumulators: np.ndarray
#     accumulator_mem: np.ndarray
#     outputs: np.ndarray

#     def __repr__(self):
#         return f"SimulationState(step={self.step})"


@dataclass
class SimulationState:
    step: int
    inputs: Dict[str, Any]
    buffer_state: Dict[str, np.ndarray]
    systolic_state: Dict[str, np.ndarray]
    accum_state: np.ndarray
    accum_outputs: np.ndarray


class MatrixEngineSimulator:
    def __init__(self, config: AcceleratorConfig):
        self.config = config
        self.history: List[SimulationState] = []
        self._setup()

    def _setup(self):
        reset_working_block()
        self.engine = MatrixEngine(self.config)
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

    def matrix_multiply(
        self, data_bank: int, weight_bank: int, accum_tile: int, accumulate: bool
    ):
        """Perform matrix multiply with current buffers and store result"""
        # Set control signals
        inputs = self._get_default_inputs()
        inputs.update(
            {
                "data_bank": data_bank,
                "weight_bank": weight_bank,
                "data_start": 1,
                "weight_start": 1,
            }
        )
        self._step(inputs)

        # Run for array_size + pipeline cycles
        inputs = self._get_default_inputs()
        for _ in range(self.config.array_size + self.config.pipeline):
            self._step(inputs)

        # Configure accumulator
        inputs.update(
            {
                "accum_tile_addr": accum_tile,
                "accum_mode": int(accumulate),
                "accum_start": 1,
            }
        )
        self._step(inputs)

        # Complete write operation
        inputs["accum_start"] = 0
        for _ in range(self.config.array_size):
            self._step(inputs)

    def read_accumulator_tile(self, tile: int) -> np.ndarray:
        """Read tile from accumulator memory"""
        inputs = self._get_default_inputs()
        inputs.update({"accum_read_tile_addr": tile, "accum_read_start": 1})
        self._step(inputs)

        # Capture outputs over array_size cycles
        results = []
        for _ in range(self.config.array_size):
            self._step()
            results.append(self.engine.get_accumulator_outputs(self.sim))

        return np.array(results[-self.config.array_size :])

    def _step(self, inputs: Dict | None = None):
        """Advance simulation and record state"""
        inputs = inputs or self._get_default_inputs()
        self.sim.step(
            {k: v for k, v in inputs.items() if not isinstance(v, WireVector)}
        )

        self.history.append(
            SimulationState(
                step=len(self.history),
                inputs=inputs,
                buffer_state=self.engine.inspect_buffer_state(self.sim),
                systolic_state=self.engine.inspect_systolic_array_state(self.sim),
                accum_state=self.engine.inspect_accumulator_state(self.sim),
                accum_outputs=self.engine.get_accumulator_outputs(self.sim),
            )
        )

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
        return systolic_state["weights"]

    @property
    def systolic_data(self) -> np.ndarray:
        """Get current data values held in the systolic array PEs.

        Returns:
            2D numpy array containing the data values currently loaded in each PE.
            Shape is (array_size, array_size).
        """
        systolic_state = self.engine.inspect_systolic_array_state(self.sim)
        return systolic_state["data"]

    @property
    def systolic_accumulators(self) -> np.ndarray:
        """Get current accumulator values in the systolic array PEs.

        Returns:
            2D numpy array containing the accumulator values in each PE.
            Shape is (array_size, array_size).
        """
        systolic_state = self.engine.inspect_systolic_array_state(self.sim)
        return systolic_state["accumulators"]

    @property
    def systolic_outputs(self) -> np.ndarray:
        """Get current output values from the systolic array PEs.

        Returns:
            2D numpy array containing the current output values from each PE.
            Shape is (array_size, array_size).
        """
        systolic_state = self.engine.inspect_systolic_array_state(self.sim)
        return systolic_state["outputs"]

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
    def print_state(self, step: int = -1):
        """Print formatted state for debugging"""
        if step >= 0:
            state = self.history[step]
            print(f"\n=== Simulation Step {state.step} ===")
            print("Inputs:", state.inputs)
            print("\nBuffer State:")
            print("Data Bank 0:\n", state.buffer_state["data_banks"][0])
            print("Weight Bank 0:\n", state.buffer_state["weight_banks"][0])
            print("\nSystolic Array:")
            print("Weights:\n", state.systolic_state["weights"])
            print("Accumulators:\n", state.systolic_state["accumulators"])
            print("\nAccumulator Outputs:", state.accum_outputs)
        else:
            print("\n=== Current Simulation State ===")
            print("\nBuffer State:")
            print("Data Bank 0:\n", self.data_banks[0])
            print("Weight Bank 0:\n", self.weight_banks[0])
            print("\nSystolic Array:")
            print("Weights:\n", self.systolic_weights)
            print("Accumulators:\n", self.systolic_accumulators)
            print("\nAccumulator Outputs:", self.accumulator_outputs)
