# tests/test_base_float.py
import pytest
from hardware_accelerators.bad_example import bar  # Import base class directly


def test_bar():
    """Test that BaseFloat cannot be instantiated directly"""
    assert bar()
