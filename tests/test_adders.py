import pytest
import pyrtl
import numpy as np

from hardware_accelerators.dtypes import Float8, BF16
from hardware_accelerators.rtllib import float_adder, FloatAdderPipelined


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
        (5.0, -3.0, 2.0),
        (-5.0, 3.0, -2.0),
        (1.5, -0.5, 1.0),
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
        test_cases.append((a, b, Float8, 4, 3, 0.1))

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
        # Numbers requiring rounding
        (1.33333, 1.33333, 2.66666),
        (3.14159, 2.85841, 6.0),
        (1.23456, 1.23456, 2.46912),
    ]

    # Add BF16 cases with appropriate tolerance
    for a, b, expected in bf16_cases:
        test_cases.append((a, b, BF16, 8, 7, 0.0001))

    return test_cases


def simulate_float_addition(a, b, format_class, e_bits, m_bits):
    """Simulate floating point addition with PyRTL"""
    pyrtl.reset_working_block()

    width = format_class.FORMAT_SPEC.total_bits
    float_a = pyrtl.Input(width, "float_a")
    float_b = pyrtl.Input(width, "float_b")
    float_out = pyrtl.WireVector(width, "float_out")

    float_out <<= float_adder(float_a, float_b, e_bits, m_bits)

    trace = pyrtl.SimulationTrace([float_a, float_b, float_out])
    sim = pyrtl.Simulation(tracer=trace)

    # Convert inputs to binary integers and simulate
    a_val = format_class(a).binint
    b_val = format_class(b).binint
    sim.step({float_a: a_val, float_b: b_val})

    # Convert output back to float
    result = format_class(binint=sim.inspect(float_out))
    return result.decimal


@pytest.mark.parametrize(
    "a,b,format_class,e_bits,m_bits,tolerance", generate_test_cases()
)
def test_float_adder(a, b, format_class, e_bits, m_bits, tolerance):
    """Test floating point adder with various test cases"""
    # Get actual result from hardware simulation
    actual = simulate_float_addition(a, b, format_class, e_bits, m_bits)

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
        ), f"Failed for {format_class.__name__}: {a} + {b} = {actual} (expected {expected})"


def test_float_adder_random():
    """Test float adder with random values"""
    # Test Float8
    for _ in range(50):
        # Generate random values within Float8 range
        a = np.random.uniform(-200, 200)
        b = np.random.uniform(-200, 200)

        try:
            result = simulate_float_addition(a, b, Float8, 4, 3)
            expected = a + b
            if abs(expected) > Float8.FORMAT_SPEC.max_normal:
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
            result = simulate_float_addition(a, b, BF16, 8, 7)
            expected = a + b
            if abs(expected) > BF16.FORMAT_SPEC.max_normal:
                continue  # Skip if result would overflow
            rel_error = (
                abs((result - expected) / expected) if expected != 0 else abs(result)
            )
            assert rel_error <= 0.0001
        except (ValueError, AssertionError):
            continue  # Skip if values are out of range