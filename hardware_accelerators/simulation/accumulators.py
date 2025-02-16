from typing import Callable, Optional, Type

import numpy as np
from pyrtl import Input, Simulation, reset_working_block

from ..dtypes import BF16, BaseFloat
from ..rtllib.accumulators import TiledAccumulatorMemoryBank
from ..rtllib.adders import float_adder


class AccumulatorBankSimulator:
    """Simulator for AccumulatorMemoryBank with integrated address generator"""

    def __init__(
        self,
        array_size: int,
        num_tiles: int,
        data_type: Type[BaseFloat] = BF16,
        adder: Callable = float_adder,
    ):
        """Initialize simulator configuration

        Args:
            array_size: Dimension of systolic array (NxN)
            num_tiles: Number of tiles to support
            data_type: Number format for data (default: BF16)
            adder: Floating point adder implementation
        """
        self.array_size = array_size
        self.num_tiles = num_tiles
        self.data_type = data_type
        self.tile_addr_width = (num_tiles - 1).bit_length()

        # Store configuration for setup
        self.config = {
            "array_size": array_size,
            "tile_addr_width": self.tile_addr_width,
            "data_type": data_type,
            "adder": adder,
        }
        self.sim = None

    def setup(self):
        """Initialize PyRTL simulation environment"""
        reset_working_block()

        # Input ports
        self._write_tile_addr = Input(self.tile_addr_width, "write_tile_addr")
        self._write_start = Input(1, "write_start")
        self._write_mode = Input(1, "write_mode")
        self._write_valid = Input(1, "write_valid")
        self._read_tile_addr = Input(self.tile_addr_width, "read_tile_addr")
        self._read_start = Input(1, "read_start")
        self._data_in = [
            Input(self.data_type.bitwidth(), f"data_in_{i}")
            for i in range(self.array_size)
        ]

        # Create accumulator bank
        self.acc_bank = TiledAccumulatorMemoryBank(**self.config)
        self.acc_bank.connect_inputs(
            self._write_tile_addr,
            self._write_start,
            self._write_mode,
            self._write_valid,
            self._read_tile_addr,
            self._read_start,
            self._data_in,  # type: ignore
        )

        # Create simulation
        self.sim = Simulation()

        return self

    def _get_default_inputs(self, updates: dict = {}) -> dict:
        """Get dictionary of default input values with optional updates"""
        defaults = {
            "write_tile_addr": 0,
            "write_start": 0,
            "write_mode": 0,
            "write_valid": 0,
            "read_tile_addr": 0,
            "read_start": 0,
            **{f"data_in_{i}": 0 for i in range(self.array_size)},
        }
        defaults.update(updates)
        return defaults

    def write_tile(
        self,
        tile_addr: int,
        data: np.ndarray,
        accumulate: bool = False,
        check_bounds: bool = True,
    ) -> None:
        """Write data to specified tile

        Args:
            tile_addr: Destination tile address
            data: Input data array (array_size x array_size)
            accumulate: If True, accumulate with existing values
            check_bounds: If True, validate input dimensions
        """
        if self.sim is None:
            raise RuntimeError("Simulator not initialized. Call setup() first")

        if check_bounds:
            if tile_addr >= self.num_tiles or tile_addr < 0:
                raise ValueError(f"Tile address {tile_addr} out of range")
            if data.shape != (self.array_size, self.array_size):
                raise ValueError(f"Data must be {self.array_size}x{self.array_size}")

        # Convert data to binary format
        binary_data = [[self.data_type(val).binint for val in row] for row in data]

        # Start write operation
        self.sim.step(
            self._get_default_inputs(
                {
                    "write_tile_addr": tile_addr,
                    "write_start": 1,
                    "write_mode": int(accumulate),
                }
            )
        )

        # Write each row
        for row in binary_data:
            self.sim.step(
                self._get_default_inputs(
                    {
                        "write_tile_addr": tile_addr,
                        "write_mode": int(accumulate),
                        "write_valid": 1,
                        **{f"data_in_{i}": row[i] for i in range(self.array_size)},
                    }
                )
            )

    def read_tile(self, tile_addr: int) -> np.ndarray:
        """Read data from specified tile

        Args:
            tile_addr: Tile address to read from

        Returns:
            Array containing tile data
        """
        if self.sim is None:
            raise RuntimeError("Simulator not initialized. Call setup() first")

        results = []

        # Start read operation
        self.sim.step(
            self._get_default_inputs({"read_tile_addr": tile_addr, "read_start": 1})
        )

        # Read rows until done
        while True:
            self.sim.step(self._get_default_inputs({"read_tile_addr": tile_addr}))

            # Capture outputs
            row = [
                float(
                    self.data_type(
                        binint=self.sim.inspect(self.acc_bank.get_output(i).name)
                    )
                )
                for i in range(self.array_size)
            ]
            results.append(row)

            if self.sim.inspect(self.acc_bank.read_done.name):
                break

        return np.array(results[-self.array_size :])

    def get_all_tiles(self) -> np.ndarray:
        """Read all tile memories

        Returns:
            3D array of shape (num_tiles, array_size, array_size)
        """
        if self.sim is None:
            raise RuntimeError("Simulator not initialized. Call setup() first")

        mems = self.acc_bank.memory_banks
        result = {}

        # Initialize empty lists for each address
        for addr in range(self.array_size * self.num_tiles):
            result[addr] = []

        # Collect memory contents
        for mem in mems:
            d = self.sim.inspect_mem(mem)
            for addr in range(self.array_size * self.num_tiles):
                result[addr].append(d.get(addr, 0))

        # Convert to numpy array and reshape
        tiles = [
            [float(self.data_type(binint=x)) for x in tile] for tile in result.values()
        ]
        tiles = np.array(tiles)

        # Reshape into tile matrices
        result_3d = []
        for i in range(self.num_tiles):
            start_idx = i * self.array_size
            end_idx = start_idx + self.array_size
            result_3d.append(tiles[start_idx:end_idx])

        return np.array(result_3d)

    def print_state(self, message: Optional[str] = None):
        """Print current state of all tiles"""
        if message:
            print(f"\n{message}")

        tiles = self.get_all_tiles()
        print("\nTile States:")
        print("-" * 50)
        for i, tile in enumerate(tiles):
            print(f"Tile {i}:")
            print(np.array2string(tile, precision=2, suppress_small=True))
        print("-" * 50)
