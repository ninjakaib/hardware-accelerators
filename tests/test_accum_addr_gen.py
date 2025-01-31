import pytest
import numpy as np
from typing import List, Dict

from hardware_accelerators.simulation.accumulators import TiledAddressGeneratorSimulator


def generate_test_sequence(
    array_size: int, tile_addr: int, write_pattern: str = "continuous"
) -> List[Dict]:
    """Generate test sequence for a single tile operation"""
    steps = []

    # Start signal
    steps.append({"tile_addr": tile_addr, "start": 1, "write_valid": 0})

    # Generate write pattern
    if write_pattern == "continuous":
        valid_pattern = [1] * array_size
    elif write_pattern == "alternating":
        valid_pattern = [1, 0] * ((array_size + 1) // 2)
    elif write_pattern == "sparse":
        rng = np.random.default_rng(42)  # Fixed seed for reproducibility
        valid_pattern = list(rng.choice([0, 1], size=array_size, p=[0.3, 0.7]))

    # Add write steps
    for valid in valid_pattern:
        steps.append({"tile_addr": tile_addr, "start": 0, "write_valid": valid})

    return steps


@pytest.mark.parametrize(
    "array_size,num_tiles",
    [
        (2, 2),  # Minimal configuration
        (4, 4),  # Small array
        (16, 8),  # Large array
    ],
)
def test_address_generator_basic(array_size: int, num_tiles: int):
    """Test basic functionality of address generator"""
    sim = TiledAddressGeneratorSimulator(array_size, num_tiles)

    # Generate test sequence for first tile
    steps = generate_test_sequence(array_size, tile_addr=0)

    # Get actual and expected behavior
    actual_history = sim.simulate(steps)
    expected_history = sim.compute_expected_behavior(steps)

    # Compare each step
    for i, (actual, expected) in enumerate(zip(actual_history, expected_history)):
        assert actual.state == expected.state, f"State mismatch at step {i}"
        assert actual.current_row == expected.row, f"Row mismatch at step {i}"
        assert actual.write_addr == expected.addr, f"Address mismatch at step {i}"
        assert (
            actual.write_enable == expected.write_enable
        ), f"Write enable mismatch at step {i}"


@pytest.mark.parametrize("write_pattern", ["continuous", "alternating", "sparse"])
def test_address_generator_patterns(write_pattern: str):
    """Test different write patterns"""
    array_size = 8
    num_tiles = 4
    sim = TiledAddressGeneratorSimulator(array_size, num_tiles)

    # Test each tile
    for tile in range(num_tiles):
        steps = generate_test_sequence(array_size, tile, write_pattern)
        actual = sim.simulate(steps)
        expected = sim.compute_expected_behavior(steps)

        # Verify behavior matches expected
        for i, (act, exp) in enumerate(zip(actual, expected)):
            assert act.state == exp.state, f"State mismatch at step {i}"
            assert act.current_row == exp.row, f"Row mismatch at step {i}"
            assert act.write_addr == exp.addr, f"Address mismatch at step {i}"
            assert (
                act.write_enable == exp.write_enable
            ), f"Write enable mismatch at step {i}"


def test_address_generator_multiple_tiles():
    """Test complex sequence with multiple tile operations"""
    array_size = 4
    num_tiles = 4
    sim = TiledAddressGeneratorSimulator(array_size, num_tiles)

    # Generate complex sequence
    steps = []
    # Complete write of tile 0
    steps.extend(generate_test_sequence(array_size, tile_addr=0))
    # Partial write of tile 2 with gaps
    steps.extend(
        generate_test_sequence(array_size, tile_addr=2, write_pattern="alternating")
    )
    # Complete write of tile 3
    steps.extend(generate_test_sequence(array_size, tile_addr=3))
    # Try to start new tile while busy
    steps.append({"tile_addr": 1, "start": 1, "write_valid": 0})

    # Get actual and expected behavior
    actual = sim.simulate(steps)
    expected = sim.compute_expected_behavior(steps)

    # Verify behavior
    for i, (act, exp) in enumerate(zip(actual, expected)):
        assert act.state == exp.state, f"State mismatch at step {i}"
        assert act.current_row == exp.row, f"Row mismatch at step {i}"
        assert act.write_addr == exp.addr, f"Address mismatch at step {i}"
        assert (
            act.write_enable == exp.write_enable
        ), f"Write enable mismatch at step {i}"
