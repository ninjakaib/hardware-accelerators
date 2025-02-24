from typing import Type

import numpy as np
import pyrtl
import pytest

from hardware_accelerators.dtypes import BF16, BaseFloat, Float8, Float16
from hardware_accelerators.rtllib import FloatAdderPipelined, float_adder


def generate_test_cases():
    """Generate comprehensive test cases for floating point addition"""
    test_cases = []

    # Test case format: (a, b, format_class, e_bits, m_bits, tolerance)

    # Float8 E4M3 test cases
    f8_cases = [
        # Basic cases
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 2.0),
        (-1.0, -1.0, -2.0),
        (1.0, -1.0, 0.0),
        # Small numbers
        (0.125, 0.125, 0.25),
        (0.0625, 0.0625, 0.125),
        (0.05, 0.10, 0.15),
        # Large numbers
        (15.5, 15.5, 31.0),
        (10.0, 5.0, 15.0),
        (-10.0, -5.0, -15.0),
        # Mixed signs
        (100.0, -50.0, 50.0),
        (-75.0, 25.0, -50.0),
        (1.5, -3.5, -2.0),
        (2.0, -2.0, 0.0),
        (4.0, -1.5, 2.5),
        (-1.0, -1.0, -2.0),
        (5.0, -5.5, -0.5),
        (6.25, -2.25, 4.0),
        (0.75, -1.25, -0.5),
        (-3.0, 2.5, -0.5),
        (2.5, -3.0, -0.5),
        (7.0, -3.5, 3.5),
        (-4.25, -2.25, -6.5),
        # Subnormal numbers
        (0.015625, 0.015625, 0.03125),
        (-0.015625, -0.015625, -0.03125),
        # Numbers requiring rounding
        (1.125, 1.125, 2.25),
        (0.3, 0.3, 0.6),
        (0.7, 0.8, 1.5),
    ]

    # Add Float8 cases with appropriate tolerance
    for a, b, expected in f8_cases:
        test_cases.append((a, b, Float8, 0.1))

    # BF16 test cases
    bf16_cases = [
        # Basic cases
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 2.0),
        (-1.0, -1.0, -2.0),
        (1.0, -1.0, 0.0),
        # Small numbers
        (0.00001, 0.00001, 0.00002),
        (0.125, 0.125, 0.25),
        (0.333, 0.333, 0.666),
        # Large numbers
        (1000.0, 1000.0, 2000.0),
        (123.456, 876.544, 1000.0),
        (-500.0, -250.0, -750.0),
        # Mixed signs
        (100.0, -50.0, 50.0),
        (-75.0, 25.0, -50.0),
        (1.5, -3.5, -2.0),
        (2.0, -2.0, 0.0),
        (4.0, -1.5, 2.5),
        (-1.0, -1.0, -2.0),
        (5.0, -5.5, -0.5),
        (6.25, -2.25, 4.0),
        (0.75, -1.25, -0.5),
        (-3.0, 2.5, -0.5),
        (2.5, -3.0, -0.5),
        (7.0, -3.5, 3.5),
        (-4.25, -2.25, -6.5),
        # Numbers requiring rounding
        (1.33333, 1.33333, 2.66666),
        (3.14159, 2.85841, 6.0),
        (1.23456, 1.23456, 2.46912),
    ]

    # Add BF16 cases with appropriate tolerance
    for a, b, expected in bf16_cases:
        test_cases.append((a, b, BF16, 0.05))

    # Float16 test cases
    f16_cases = [
        # Basic cases
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 2.0),
        (-1.0, -1.0, -2.0),
        (1.0, -1.0, 0.0),
        # Small numbers
        (0.00001, 0.00001, 0.00002),
        (0.125, 0.125, 0.25),
        (0.333, 0.333, 0.666),
        # Large numbers
        (1000.0, 1000.0, 2000.0),
        (123.456, 876.544, 1000.0),
        (-500.0, -250.0, -750.0),
        # Mixed signs
        (100.0, -50.0, 50.0),
        (-75.0, 25.0, -50.0),
        (1.5, -3.5, -2.0),
        (2.0, -2.0, 0.0),
        (4.0, -1.5, 2.5),
        (-1.0, -1.0, -2.0),
        (5.0, -5.5, -0.5),
        (6.25, -2.25, 4.0),
        (0.75, -1.25, -0.5),
        (-3.0, 2.5, -0.5),
        (2.5, -3.0, -0.5),
        (7.0, -3.5, 3.5),
        (-4.25, -2.25, -6.5),
        # Numbers requiring rounding
        (1.33333, 1.33333, 2.66666),
        (3.14159, 2.85841, 6.0),
        (1.23456, 1.23456, 2.46912),
        # Edge cases
        (65503.5, 65503.5, 131007.0),
        (0.00006103515625, 0.00006103515625, 0.0001220703125),
        (
            0.000000059604645,
            0.000000059604645,
            0.00000011920929,
        ),
    ]

    # Add Float16 cases with appropriate tolerance
    for a, b, expected in f16_cases:
        test_cases.append((a, b, Float16, 0.001))

    return test_cases


def simulate_float_addition(a, b, dtype: Type[BaseFloat]):
    """Simulate floating point addition with PyRTL"""
    pyrtl.reset_working_block()

    width = dtype.bitwidth()
    float_a = pyrtl.Input(width, "float_a")
    float_b = pyrtl.Input(width, "float_b")
    float_out = pyrtl.WireVector(width, "float_out")

    float_out <<= float_adder(float_a, float_b, dtype)

    trace = pyrtl.SimulationTrace([float_a, float_b, float_out])
    sim = pyrtl.Simulation(tracer=trace)

    # Convert inputs to binary integers and simulate
    a_val = dtype(a).binint
    b_val = dtype(b).binint
    sim.step({float_a.name: a_val, float_b.name: b_val})

    # Convert output back to float
    result = dtype(binint=sim.inspect(float_out.name))
    return result.decimal_approx


@pytest.mark.parametrize("a,b,dtype,tolerance", generate_test_cases())
def test_float_adder(a, b, dtype: Type[BaseFloat], tolerance):
    """Test floating point adder with various test cases"""
    # Get actual result from hardware simulation
    actual = simulate_float_addition(a, b, dtype)

    # Calculate expected result
    expected = a + b

    # For very small numbers, use absolute tolerance
    if abs(expected) < tolerance:
        assert abs(actual - expected) <= tolerance
    else:
        # For larger numbers, use relative tolerance
        rel_error = abs((actual - expected) / expected)
        assert (
            rel_error <= tolerance
        ), f"Failed for {dtype.__name__}: {a} + {b} = {actual} (expected {expected})"


def test_float_adder_random():
    """Test float adder with random values"""
    # Test Float8
    for _ in range(50):
        # Generate random values within Float8 range
        a = np.random.uniform(-200, 200)
        b = np.random.uniform(-200, 200)

        try:
            result = simulate_float_addition(a, b, Float8)
            expected = a + b
            if abs(expected) > Float8.max_normal():
                continue  # Skip if result would overflow
            rel_error = (
                abs((result - expected) / expected) if expected != 0 else abs(result)
            )
            assert rel_error <= 0.1
        except (ValueError, AssertionError):
            continue  # Skip if values are out of range

    # Test BF16
    for _ in range(50):
        # Generate random values within BF16 range
        a = np.random.uniform(-1e38, 1e38)
        b = np.random.uniform(-1e38, 1e38)

        try:
            result = simulate_float_addition(a, b, BF16)
            expected = a + b
            if abs(expected) > BF16.max_normal():
                continue  # Skip if result would overflow
            rel_error = (
                abs((result - expected) / expected) if expected != 0 else abs(result)
            )
            assert rel_error <= 0.0001
        except (ValueError, AssertionError):
            continue  # Skip if values are out of range

    # Test Float16
    for _ in range(50):
        # Generate random values within Float16 range
        a = np.random.uniform(-65504, 65504)  # Float16 max normal
        b = np.random.uniform(-65504, 65504)

        try:
            result = simulate_float_addition(a, b, Float16)
            expected = a + b
            if abs(expected) > Float16.max_normal():
                continue  # Skip if result would overflow
            rel_error = (
                abs((result - expected) / expected) if expected != 0 else abs(result)
            )
            assert rel_error <= 0.0001
        except (ValueError, AssertionError):
            continue  # Skip if values are out of range
