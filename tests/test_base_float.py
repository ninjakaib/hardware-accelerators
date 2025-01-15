# tests/test_base_float.py
import pytest
from hardware_accelerators.dtypes import BaseFloat  # Import base class directly


def test_base_float_abstract():
    """Test that BaseFloat cannot be instantiated directly"""
    with pytest.raises(TypeError):
        BaseFloat(1.0)
