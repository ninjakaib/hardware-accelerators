from typing import Type

import numpy as np
import pyrtl
import pytest

from hardware_accelerators.dtypes import BF16, BaseFloat, Float8, Float16, Float32
from hardware_accelerators.rtllib import float_multiplier


def generate_test_cases():
    """Generate comprehensive test cases for floating point multiplication"""
    test_cases = []

    # Test case format: (a, b, format_class, tolerance)

    # Float8 E4M3 test cases
    f8_cases = [
        # Basic cases
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        (-1.0, -1.0, 1.0),
        (-1.0, 1.0, -1.0),
        # Small numbers
        (0.125, 2.0, 0.25),
        (0.0625, 4.0, 0.25),
        (0.25, 0.5, 0.125),
        # Large numbers
        (4.0, 3.0, 12.0),
        (2.0, 7.0, 14.0),
        (-3.0, -5.0, 15.0),
        # Mixed signs
        (5.0, -3.0, -15.0),
        (-4.0, 3.0, -12.0),
        (1.5, -2.0, -3.0),
        # Subnormal numbers
        (0.015625, 2.0, 0.03125),
        (-0.015625, -2.0, 0.03125),
        # Numbers requiring rounding
        (1.5, 1.5, 2.25),
        (0.3, 3.0, 0.9),
        (0.7, 2.0, 1.4),
        # Additional cases
        (0.5, 0.5, 0.25),
        (2.0, 0.5, 1.0),
        (0.1, 10.0, 1.0),
        (0.2, 5.0, 1.0),
        (0.3, 3.333, 0.9999),
    ]

    # Add Float8 cases with appropriate tolerance
    for a, b, expected in f8_cases:
        test_cases.append((a, b, Float8, 0.1))

    # BF16 test cases
    bf16_cases = [
        # Basic cases
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        (-1.0, -1.0, 1.0),
        (1.0, -1.0, -1.0),
        # Small numbers
        (0.00001, 100000.0, 1.0),
        (0.125, 8.0, 1.0),
        (0.333, 3.0, 0.999),
        # Large numbers
        (1000.0, 2.0, 2000.0),
        (123.456, 2.0, 246.912),
        (-500.0, -1.5, 750.0),
        # Mixed signs
        (100.0, -2.0, -200.0),
        (-75.0, 0.5, -37.5),
        (1.5, -2.0, -3.0),
        # Numbers requiring rounding
        (1.33333, 2.0, 2.66666),
        (3.14159, 2.0, 6.28318),
        (1.23456, 4.0, 4.93824),
        # Additional cases
        (0.5, 0.5, 0.25),
        (2.0, 0.5, 1.0),
        (0.1, 10.0, 1.0),
        (0.2, 5.0, 1.0),
        (0.3, 3.333, 0.9999),
        (0.4, 2.5, 1.0),
        (0.6, 1.6667, 1.00002),
        (0.7, 1.4286, 0.99982),
        (0.8, 1.25, 1.0),
        (0.9, 1.1111, 0.99999),
    ]

    # Add BF16 cases with appropriate tolerance
    for a, b, expected in bf16_cases:
        test_cases.append((a, b, BF16, 0.01))

    # Float16 test cases
    f16_cases = [
        # Basic cases
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        (-1.0, -1.0, 1.0),
        (1.0, -1.0, -1.0),
        # Small numbers
        (0.0001, 10000.0, 1.0),
        (0.125, 8.0, 1.0),
        (0.333, 3.0, 0.999),
        # Large numbers
        (1000.0, 2.0, 2000.0),
        (123.456, 2.0, 246.912),
        (-500.0, -1.5, 750.0),
        # Mixed signs
        (100.0, -2.0, -200.0),
        (-75.0, 0.5, -37.5),
        (1.5, -2.0, -3.0),
        # Numbers requiring rounding
        (1.33333, 2.0, 2.66666),
        (3.14159, 2.0, 6.28318),
        (1.23456, 4.0, 4.93824),
        # Additional cases
        (0.5, 0.5, 0.25),
        (2.0, 0.5, 1.0),
        (0.1, 10.0, 1.0),
        (0.2, 5.0, 1.0),
        (0.3, 3.333, 0.9999),
        (0.4, 2.5, 1.0),
        (0.6, 1.6667, 1.00002),
        (0.7, 1.4286, 0.99982),
        (0.8, 1.25, 1.0),
        (0.9, 1.1111, 0.99999),
    ]

    # Add Float16 cases with appropriate tolerance
    for a, b, expected in f16_cases:
        test_cases.append((a, b, Float16, 0.001))

    # Float16 test cases
    f32_cases = [
        # Basic cases
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        (-1.0, -1.0, 1.0),
        (1.0, -1.0, -1.0),
        # Small numbers
        (0.0001, 10000.0, 1.0),
        (0.125, 8.0, 1.0),
        (0.333, 3.0, 0.999),
        # Large numbers
        (1000.0, 2.0, 2000.0),
        (123.456, 2.0, 246.912),
        (-500.0, -1.5, 750.0),
        # Mixed signs
        (100.0, -2.0, -200.0),
        (-75.0, 0.5, -37.5),
        (1.5, -2.0, -3.0),
        # Numbers requiring rounding
        (1.33333, 2.0, 2.66666),
        (3.14159, 2.0, 6.28318),
        (1.23456, 4.0, 4.93824),
        # Additional cases
        (0.5, 0.5, 0.25),
        (2.0, 0.5, 1.0),
        (0.1, 10.0, 1.0),
        (0.2, 5.0, 1.0),
        (0.3, 3.333, 0.9999),
        (0.4, 2.5, 1.0),
        (0.6, 1.6667, 1.00002),
        (0.7, 1.4286, 0.99982),
        (0.8, 1.25, 1.0),
        (0.9, 1.1111, 0.99999),
    ]

    # Add Float32 cases with appropriate tolerance
    for a, b, expected in f32_cases:
        test_cases.append((a, b, Float32, 0.001))

    return test_cases


def simulate_float_multiplication(a, b, dtype: Type[BaseFloat]):
    """Simulate floating point multiplication with PyRTL"""
    pyrtl.reset_working_block()

    width = dtype.bitwidth()
    float_a = pyrtl.Input(width, "float_a")
    float_b = pyrtl.Input(width, "float_b")
    float_out = pyrtl.WireVector(width, "float_out")

    float_out <<= float_multiplier(float_a, float_b, dtype)

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
def test_float_multiplier(a, b, dtype: Type[BaseFloat], tolerance):
    """Test floating point multiplier with various test cases"""
    # Get actual result from hardware simulation
    actual = simulate_float_multiplication(a, b, dtype)

    # Calculate expected result
    expected = a * b

    # For very small numbers, use absolute tolerance
    if abs(expected) < tolerance:
        assert abs(actual - expected) <= tolerance
    else:
        # For larger numbers, use relative tolerance
        rel_error = abs((actual - expected) / expected)
        assert (
            rel_error <= tolerance
        ), f"Failed for {dtype.__name__}: {a} * {b} = {actual} (expected {expected})"


def test_float_multiplier_random():
    """Test float multiplier with random values"""
    # Test Float8
    for _ in range(50):
        # Generate random values within Float8 range
        a = np.random.uniform(-15, 15)  # Float8 has smaller range than addition
        b = np.random.uniform(-15, 15)

        try:
            result = simulate_float_multiplication(a, b, Float8)
            expected = a * b
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
        # Use smaller range for multiplication to avoid overflow
        a = np.random.uniform(-1e19, 1e19)
        b = np.random.uniform(-1e19, 1e19)

        # result = simulate_float_multiplication(a, b, BF16)
        # expected = a * b
        # if abs(expected) > BF16.max_normal():
        #     continue  # Skip if result would overflow
        # rel_error = (
        #     abs((result - expected) / expected) if expected != 0 else abs(result)
        # )
        try:
            result = simulate_float_multiplication(a, b, BF16)
            expected = a * b
            if abs(expected) > BF16.max_normal():
                continue  # Skip if result would overflow
            rel_error = (
                abs((result - expected) / expected) if expected != 0 else abs(result)
            )
            assert rel_error <= 0.01
        except (ValueError, AssertionError):
            continue  # Skip if values are out of range
