from typing import Dict, List

import numpy as np
import pytest

from hardware_accelerators.dtypes import BF16, Float8
from hardware_accelerators.simulation.accumulators import AccumulatorBankSimulator


def basic_test_accumulator():
    # Create and setup simulator
    sim = AccumulatorBankSimulator(array_size=3, num_tiles=4).setup()

    # Test data
    test_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])

    # Initial state
    sim.print_state("Initial State")

    # Write to tile 0
    sim.write_tile(0, test_data)
    sim.print_state("After Write to Tile 0")

    # Write to tile 2
    sim.write_tile(2, test_data)
    sim.print_state("After Write to Tile 2")

    # Accumulate into tile 0
    sim.write_tile(0, test_data, accumulate=True)
    sim.print_state("After Accumulation to Tile 0")

    # Validate results
    tile0_data = sim.read_tile(0)
    tile2_data = sim.read_tile(2)

    expected_tile0 = np.array([[2, 4, 6], [8, 10, 12], [14, 16, 18]])
    expected_tile2 = test_data

    np.testing.assert_allclose(tile0_data, expected_tile0)
    np.testing.assert_allclose(tile2_data, expected_tile2)
    print("All tests passed!")


@pytest.fixture
def basic_simulator():
    """Basic 3x3 simulator with 4 tiles"""
    return AccumulatorBankSimulator(array_size=3, num_tiles=4).setup()


@pytest.fixture
def large_simulator():
    """Larger 8x8 simulator with 8 tiles"""
    return AccumulatorBankSimulator(array_size=8, num_tiles=8).setup()


@pytest.fixture
def float8_simulator():
    """3x3 simulator using Float8 instead of BF16"""
    return AccumulatorBankSimulator(array_size=3, num_tiles=4, data_type=Float8).setup()


def get_test_data(size: int) -> np.ndarray:
    """Generate test data matrix of specified size"""
    return np.arange(1, size * size + 1).reshape((size, size)).astype(float)


def assert_tile_equal(
    actual: np.ndarray, expected: np.ndarray, tile_num: int, rtol: float = 1e-5
):
    """Assert two tiles are equal with detailed error message"""
    np.testing.assert_allclose(
        actual,
        expected,
        rtol=rtol,
        err_msg=f"\nTile {tile_num} mismatch:"
        f"\nExpected:\n{expected}"
        f"\nGot:\n{actual}"
        f"\nDiff:\n{expected - actual}",
    )


class TestAccumulatorBankSimulator:
    """Test suite for AccumulatorBankSimulator"""

    def test_initialization(self, basic_simulator):
        """Test simulator initialization and configuration"""
        assert basic_simulator.array_size == 3
        assert basic_simulator.num_tiles == 4
        assert basic_simulator.data_type == BF16
        assert basic_simulator.tile_addr_width == 2
        assert basic_simulator.sim is not None
        assert basic_simulator.acc_bank is not None

    def test_uninitialized_error(self):
        """Test error handling when simulator not initialized"""
        sim = AccumulatorBankSimulator(array_size=3, num_tiles=4)
        with pytest.raises(RuntimeError, match="Simulator not initialized"):
            sim.write_tile(0, np.zeros((3, 3)))

    @pytest.mark.parametrize(
        "tile_addr,error_type,match",
        [
            (4, ValueError, "Tile address 4 out of range"),
            (-1, ValueError, "Tile address -1 out of range"),
        ],
    )
    def test_invalid_tile_address(self, basic_simulator, tile_addr, error_type, match):
        """Test error handling for invalid tile addresses"""
        with pytest.raises(error_type, match=match):
            basic_simulator.write_tile(tile_addr, np.zeros((3, 3)))

    @pytest.mark.parametrize(
        "data_shape,error_msg",
        [
            ((2, 2), "Data must be 3x3"),
            ((3, 4), "Data must be 3x3"),
            ((4, 3), "Data must be 3x3"),
        ],
    )
    def test_invalid_data_shape(self, basic_simulator, data_shape, error_msg):
        """Test error handling for incorrect input data shapes"""
        with pytest.raises(ValueError, match=error_msg):
            basic_simulator.write_tile(0, np.zeros(data_shape))

    def test_single_tile_write(self, basic_simulator):
        """Test basic write to single tile"""
        test_data = get_test_data(3)
        basic_simulator.write_tile(0, test_data)

        # Read back and verify
        result = basic_simulator.read_tile(0)
        assert_tile_equal(result, test_data, 0)

        # Check other tiles are zero
        all_tiles = basic_simulator.get_all_tiles()
        for i in range(1, 4):
            assert_tile_equal(all_tiles[i], np.zeros((3, 3)), i, rtol=1e-6)

    def test_multiple_tile_writes(self, basic_simulator):
        """Test writing to multiple tiles"""
        test_data = get_test_data(3)

        # Write to alternating tiles
        basic_simulator.write_tile(0, test_data)
        basic_simulator.write_tile(2, test_data * 2)

        # Verify tile 0
        result0 = basic_simulator.read_tile(0)
        assert_tile_equal(result0, test_data, 0)

        # Verify tile 2
        result2 = basic_simulator.read_tile(2)
        assert_tile_equal(result2, test_data * 2, 2)

        # Verify tiles 1 and 3 are zero
        all_tiles = basic_simulator.get_all_tiles()
        for i in [1, 3]:
            assert_tile_equal(all_tiles[i], np.zeros((3, 3)), i, rtol=1e-6)

    def test_accumulation(self, basic_simulator):
        """Test accumulation mode"""
        test_data = get_test_data(3)

        # Write initial data
        basic_simulator.write_tile(1, test_data)

        # Accumulate same data
        basic_simulator.write_tile(1, test_data, accumulate=True)

        # Verify result is doubled
        result = basic_simulator.read_tile(1)
        assert_tile_equal(result, test_data * 2, 1)

    def test_overwrite(self, basic_simulator):
        """Test overwrite behavior"""
        test_data = get_test_data(3)

        # Write initial data
        basic_simulator.write_tile(1, test_data)

        # Overwrite with new data
        new_data = test_data * 3
        basic_simulator.write_tile(1, new_data)

        # Verify overwrite
        result = basic_simulator.read_tile(1)
        assert_tile_equal(result, new_data, 1)

    def test_complex_sequence(self, basic_simulator):
        """Test complex sequence of operations"""
        test_data = get_test_data(3)

        # Sequence of operations
        ops = [
            (0, test_data, False),  # Write to tile 0
            (2, test_data * 2, False),  # Write to tile 2
            (0, test_data, True),  # Accumulate in tile 0
            (2, test_data, True),  # Accumulate in tile 2
            (1, test_data * 3, False),  # Write to tile 1
        ]

        # Execute operations
        for tile_addr, data, accumulate in ops:
            basic_simulator.write_tile(tile_addr, data, accumulate)

        # Verify final state
        expected = [
            test_data * 2,  # Tile 0: Original + Accumulation
            test_data * 3,  # Tile 1: Single write
            test_data * 3,  # Tile 2: Original*2 + Accumulation
            np.zeros((3, 3)),  # Tile 3: Unused
        ]

        all_tiles = basic_simulator.get_all_tiles()
        for i, exp in enumerate(expected):
            assert_tile_equal(all_tiles[i], exp, i)

    def test_large_array(self, large_simulator):
        """Test operations with larger array size"""
        test_data = get_test_data(8)

        # Write to multiple tiles
        large_simulator.write_tile(0, test_data)
        large_simulator.write_tile(7, test_data * 2)

        # Verify extremes
        result0 = large_simulator.read_tile(0)
        result7 = large_simulator.read_tile(7)

        assert_tile_equal(result0, test_data, 0)
        assert_tile_equal(result7, test_data * 2, 7)

    def test_float8_precision(self, float8_simulator):
        """Test behavior with reduced precision Float8"""
        test_data = get_test_data(3)

        # Write and accumulate
        float8_simulator.write_tile(0, test_data)
        float8_simulator.write_tile(0, test_data, accumulate=True)

        # Result should show reduced precision
        result = float8_simulator.read_tile(0)

        # Use larger tolerance for Float8
        assert_tile_equal(result, test_data * 2, 0, rtol=1e-2)

    @pytest.mark.parametrize("size,num_tiles", [(2, 2), (4, 4), (6, 8), (8, 16)])
    def test_different_configurations(self, size, num_tiles):
        """Test different array size and tile count configurations"""
        sim = AccumulatorBankSimulator(array_size=size, num_tiles=num_tiles).setup()

        test_data = get_test_data(size)

        # Write to first and last tile
        sim.write_tile(0, test_data)
        sim.write_tile(num_tiles - 1, test_data * 2)

        # Verify
        result0 = sim.read_tile(0)
        result_last = sim.read_tile(num_tiles - 1)

        assert_tile_equal(result0, test_data, 0)
        assert_tile_equal(result_last, test_data * 2, num_tiles - 1)
