from typing import List, Type
from pyrtl import Input, reset_working_block, Simulation
import numpy as np

from .utils import convert_array_dtype

from .matrix_utils import chunk_weight_matrix
from ..rtllib import WeightFIFO, BufferMemory
from ..dtypes import *


from typing import List, Type, Dict, Optional
from pyrtl import Input, reset_working_block, Simulation
import numpy as np
from ..rtllib import WeightFIFO
from ..dtypes import BaseFloat, BF16


class WeightFIFOSimulator:
    """Simulator for WeightFIFO that manages tiled weight storage and streaming"""

    def __init__(
        self,
        array_size: int,
        num_tiles: int,
        dtype: Type[BaseFloat] = BF16,
    ):
        """Initialize simulator configuration

        Args:
            array_size: Dimension of systolic array (NxN)
            num_tiles: Number of weight tiles to store
            dtype: Hardware floating point format for weights (default: BF16)
        """
        self.array_size = array_size
        self.num_tiles = num_tiles
        self.dtype = dtype

        # Store configuration
        self.config = {
            "array_size": array_size,
            "num_tiles": num_tiles,
            "dtype": dtype,
        }

    def setup(self):
        """Initialize PyRTL simulation environment"""
        reset_working_block()

        # Create weight FIFO
        self.fifo = WeightFIFO(**self.config)

        # Input control ports
        self._start = Input(1, "start")
        self._tile_addr = Input(self.fifo.tile_addr_width, "tile_addr")
        self.fifo.connect_inputs(start=self._start, tile_addr=self._tile_addr)

        # Create simulation
        self.sim = Simulation()
        return self

    def _get_default_inputs(self, updates: dict = {}) -> dict:
        """Get dictionary of default input values with optional updates"""
        defaults = {
            "start": 0,
            "tile_addr": 0,
        }
        defaults.update(updates)
        return defaults

    def vec_to_binary(self, vec):
        """Convert vector to concatenated binary representation"""
        concatenated = 0
        for i, d in enumerate(vec[::-1]):
            binary = self.dtype(d).binint
            concatenated += binary << (i * self.dtype.bitwidth())
        return concatenated

    def load_weights(self, weights: np.ndarray, check_bounds: bool = True) -> None:
        """Load weights into FIFO memory

        Args:
            weights: Either:
                    - Dictionary mapping tile addresses to weight matrices
                    - Single weight matrix to be automatically tiled
            check_bounds: Whether to validate matrix dimensions

        Raises:
            ValueError: If matrix dimensions are invalid
            RuntimeError: If simulator not initialized
        """
        # if not hasattr(self, "sim"):
        self.setup()

        weights = chunk_weight_matrix(weights, self.array_size)
        # binary_weights = convert_array_dtype(weights, self.dtype)

        # Load tiles into memory
        memory = self.sim.inspect_mem(self.fifo.memory)
        for tile_addr, tile_matrix in enumerate(weights):
            if check_bounds and tile_matrix.shape != (self.array_size, self.array_size):
                raise ValueError(
                    f"Tile at address {tile_addr} has invalid shape {tile_matrix.shape}"
                )
            base_addr = tile_addr * self.array_size
            for i, row in enumerate(tile_matrix[::-1]):
                memory[base_addr + i] = self.vec_to_binary(row)

    def read_tile(self, tile_addr: int) -> np.ndarray:
        """Stream weights from a specific tile

        Args:
            tile_addr: Address of tile to read

        Returns:
            List of weight vectors streamed from the tile

        Raises:
            RuntimeError: If simulator not initialized
            ValueError: If tile_addr is invalid
        """
        if self.sim is None:
            raise RuntimeError("Simulator not initialized. Call setup() first")

        if tile_addr >= self.num_tiles:
            raise ValueError(f"Invalid tile address {tile_addr}")

        weight_vectors = np.zeros((self.array_size, self.array_size))

        # Start streaming
        self.sim.step(
            self._get_default_inputs(
                {
                    "start": 1,
                    "tile_addr": tile_addr,
                }
            )
        )

        # Stream for array_size cycles
        for i in range(self.array_size):
            self.sim.step(self._get_default_inputs())

            # Capture outputs
            weight_vec = [
                float(self.dtype(binint=self.sim.inspect(wire.name)))
                for wire in self.fifo.weights_out
            ]
            weight_vectors[self.array_size - 1 - i] = weight_vec

        return weight_vectors


class BufferMemorySimulator:
    """Simulator for BufferMemory with dual banks for data and weights"""

    def __init__(
        self,
        array_size: int,
        data_type: Type[BaseFloat] = BF16,
        weight_type: Type[BaseFloat] = BF16,
    ):
        """Initialize simulator configuration

        Args:
            array_size: Dimension of systolic array (NxN)
            data_type: Number format for data (default: BF16)
            weight_type: Number format for weights (default: BF16)
        """
        self.array_size = array_size
        self.data_type = data_type
        self.weight_type = weight_type

        # Store configuration
        self.config = {
            "array_size": array_size,
            "data_type": data_type,
            "weight_type": weight_type,
        }
        self.sim = None

    def setup(self):
        """Initialize PyRTL simulation environment"""
        reset_working_block()

        # Input control ports
        self._data_start = Input(1, "data_start")
        self._data_select = Input(1, "data_select")
        self._weight_start = Input(1, "weight_start")
        self._weight_select = Input(1, "weight_select")

        # Create buffer memory
        self.buffer = BufferMemory(**self.config)
        self.buffer.connect_inputs(
            self._data_start, self._data_select, self._weight_start, self._weight_select
        )

        # Create simulation
        self.sim = Simulation()
        return self

    def _get_default_inputs(self, updates: dict = {}) -> dict:
        """Get dictionary of default input values with optional updates"""
        defaults = {
            "data_start": 0,
            "data_select": 0,
            "weight_start": 0,
            "weight_select": 0,
        }
        defaults.update(updates)
        return defaults

    def vec_to_binary(self, vec, dtype: Type[BaseFloat]):
        """Convert vector to concatenated binary representation"""
        concatenated = 0
        for i, d in enumerate(vec[::-1]):
            binary = dtype(d).binint
            concatenated += binary << (i * dtype.bitwidth())
        return concatenated

    def load_memories(
        self,
        data_bank: int | None = None,
        weight_bank: int | None = None,
        data: np.ndarray | None = None,
        weights: np.ndarray | None = None,
        check_bounds: bool = True,
    ) -> None:
        """Load data and weights into specified memory banks"""
        if self.sim is None:
            raise RuntimeError("Simulator not initialized. Call setup() first")

        # Convert matrices to binary format
        if data is not None:
            assert data_bank is not None
            if check_bounds and data.shape != (self.array_size, self.array_size):
                raise ValueError(f"Data must be {self.array_size}x{self.array_size}")
            data_mem = self.sim.inspect_mem(self.buffer.data_mems[data_bank])
            # Load directly into memory banks
            for i, row in enumerate(data[::-1]):
                data_mem[i] = self.vec_to_binary(row, self.data_type)

        if weights is not None:
            assert weight_bank is not None
            if check_bounds and weights.shape != (self.array_size, self.array_size):
                raise ValueError(f"Weights must be {self.array_size}x{self.array_size}")
            weight_mem = self.sim.inspect_mem(self.buffer.weight_mems[weight_bank])
            # Load directly into memory banks
            for i, row in enumerate(weights[::-1]):
                weight_mem[i] = self.vec_to_binary(row, self.weight_type)

    def stream_data(
        self,
        data_bank: int,
    ) -> List[np.ndarray]:
        """Stream data from memory to systolic array

        Args:
            data_bank: Data memory bank to read from

        Returns:
            List of data vectors streamed to systolic array
        """
        if self.sim is None:
            raise RuntimeError("Simulator not initialized. Call setup() first")

        data_vectors = []

        # Start streaming
        self.sim.step(
            self._get_default_inputs(
                {
                    "data_start": 1,
                    "data_select": data_bank,
                }
            )
        )

        # Stream for array_size cycles
        for _ in range(self.array_size):
            self.sim.step(
                self._get_default_inputs(
                    {
                        "data_select": data_bank,
                    }
                )
            )

            # Capture outputs
            data_vec = [
                float(self.data_type(binint=self.sim.inspect(wire.name)))
                for wire in self.buffer.datas_out
            ]
            data_vectors.insert(0, data_vec)

        return data_vectors

    def stream_weights(
        self,
        weight_bank: int,
    ) -> List[np.ndarray]:
        """Stream weights from memory to systolic array

        Args:
            weight_bank: Weight memory bank to read from

        Returns:
            List of weight vectors streamed to systolic array
        """
        if self.sim is None:
            raise RuntimeError("Simulator not initialized. Call setup() first")

        weight_vectors = []

        # Start streaming
        self.sim.step(
            self._get_default_inputs(
                {
                    "weight_start": 1,
                    "weight_select": weight_bank,
                }
            )
        )

        # Stream for array_size cycles
        for _ in range(self.array_size):
            self.sim.step(
                self._get_default_inputs(
                    {
                        "weight_select": weight_bank,
                    }
                )
            )

            # Capture outputs
            weight_vec = [
                float(self.weight_type(binint=self.sim.inspect(wire.name)))
                for wire in self.buffer.weights_out
            ]
            weight_vectors.insert(0, weight_vec)

        return weight_vectors
