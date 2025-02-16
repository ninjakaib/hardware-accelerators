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

from ..rtllib.multipliers import float_multiplier
from ..dtypes import BaseFloat, BF16
from ..rtllib.accumulators import TiledAccumulatorMemoryBank
from ..rtllib.adders import float_adder
from ..rtllib.systolic import SystolicArrayDiP, SystolicArraySimState
from ..rtllib.buffer import BufferMemory
from ..rtllib import TiledAcceleratorConfig, TiledMatrixEngine
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
        self.history: List[SimulationState] = []
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
            SimulationState(
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


class AcceleratorSimulator:
    pass
