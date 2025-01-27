# tests/conftest.py
import pytest

from hardware_accelerators import BF16, Float8


# Add any shared fixtures here
@pytest.fixture
def sample_float8_values():
    """Fixture providing common Float8 test values"""
    return [
        (1.0, "0.0111.000"),
        (0.5, "0.0110.000"),
        (-1.0, "1.0111.000"),
        (0.0, "0.0000.000"),
    ]


@pytest.fixture
def sample_bfloat16_values():
    """Fixture providing common BF16 test values"""
    return [
        (1.0, "0011111110000000"),
        (2.0, "0100000000000000"),
        (-1.0, "1011111110000000"),
        (0.0, "0000000000000000"),
    ]
