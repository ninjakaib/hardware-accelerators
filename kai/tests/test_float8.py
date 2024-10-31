import pytest
from float8 import decimal_to_e4m3
import math

# Fixture for commonly used values
@pytest.fixture
def edge_cases():
    return {
        'MAX_NORMAL': 448.0,
        'MIN_NORMAL': 2**-6,
        'MAX_SUBNORMAL': 0.875 * 2**-6,
        'MIN_SUBNORMAL': 2**-9
    }

def test_special_values():
    assert decimal_to_e4m3(0) == "00000000"
    assert decimal_to_e4m3(-0) == "00000000"
    assert decimal_to_e4m3(float('nan')) == "01111111"

def test_edge_cases(edge_cases):
    # Positive edge cases
    assert decimal_to_e4m3(edge_cases['MAX_NORMAL']) == "01111110"
    assert decimal_to_e4m3(edge_cases['MIN_NORMAL']) == "00001000"
    assert decimal_to_e4m3(edge_cases['MAX_SUBNORMAL']) == "00000111"
    assert decimal_to_e4m3(edge_cases['MIN_SUBNORMAL']) == "00000001"
    
    # Negative edge cases
    assert decimal_to_e4m3(-edge_cases['MAX_NORMAL']) == "11111110"
    assert decimal_to_e4m3(-edge_cases['MIN_NORMAL']) == "10001000"
    assert decimal_to_e4m3(-edge_cases['MAX_SUBNORMAL']) == "10000111"
    assert decimal_to_e4m3(-edge_cases['MIN_SUBNORMAL']) == "10000001"

def test_normal_numbers():
    test_cases = [
        (1.0, "00111000"),
        (-1.0, "10111000"),
        (2.0, "01000000"),
        (0.5, "00110000"),
        (0.25, "00101000")
    ]
    
    for value, expected in test_cases:
        assert decimal_to_e4m3(value) == expected

def test_numbers_requiring_rounding():
    test_cases = [
        (1.75, "00111110"),
        (1.25, "00111010")
    ]
    
    for value, expected in test_cases:
        assert decimal_to_e4m3(value) == expected

def test_boundary_values():
    assert decimal_to_e4m3(448) == "01111110"
    assert decimal_to_e4m3(2**-6 + 2**-9) == "00001001"

def test_subnormal_numbers():
    test_cases = [
        (0.5 * 2**-6, "00000100"),
        (0.25 * 2**-6, "00000010")
    ]
    
    for value, expected in test_cases:
        assert decimal_to_e4m3(value) == expected

# Parametrized test example
@pytest.mark.parametrize("input_value,expected", [
    (1.0, "00111000"),
    (-1.0, "10111000"),
    (0.0, "00000000"),
    (2.0, "01000000"),
    (0.5, "00110000"),
])
def test_parametrized_values(input_value, expected):
    assert decimal_to_e4m3(input_value) == expected
