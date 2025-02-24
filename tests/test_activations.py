import pytest
import numpy as np
from hardware_accelerators.dtypes import BF16, Float8, Float16
from hardware_accelerators.simulation.activations import ReluSimulator


@pytest.mark.parametrize("dtype", [BF16, Float8, Float16])
def test_relu_basic(dtype):
    """Test basic ReLU functionality"""
    sim = ReluSimulator(dtype)

    # Test single value as 1D array
    assert np.isclose(sim.activate(np.array([1.0])), [1.0])
    assert np.isclose(sim.activate(np.array([-1.0])), [0.0])
    assert np.isclose(sim.activate(np.array([0.0])), [0.0])


@pytest.mark.parametrize("dtype", [BF16, Float8, Float16])
def test_relu_vector(dtype):
    """Test ReLU with vector inputs"""
    sim = ReluSimulator(dtype)

    input_vector = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    expected = np.array([0.0, 0.0, 0.0, 1.0, 2.0])

    result = sim.activate(input_vector)
    assert np.allclose(result, expected, rtol=0.01)


@pytest.mark.parametrize("dtype", [BF16, Float8, Float16])
def test_relu_matrix(dtype):
    """Test ReLU with matrix inputs"""
    sim = ReluSimulator(dtype)

    input_matrix = np.array([[-1.0, 2.0, -3.0], [4.0, -5.0, 6.0], [-7.0, 8.0, -9.0]])
    expected = np.array([[0.0, 2.0, 0.0], [4.0, 0.0, 6.0], [0.0, 8.0, 0.0]])

    result = sim.activate(input_matrix)
    assert np.allclose(result, expected, rtol=0.01)


@pytest.mark.parametrize("dtype", [BF16, Float8, Float16])
def test_relu_enable_timing(dtype):
    """Test that enable signal properly latches"""
    sim = ReluSimulator(dtype)

    # Test that enable takes effect after one cycle
    input_matrix = np.array([[-1.0, -2.0], [-3.0, -4.0]])

    # With enable=True, should get zeros
    result1 = sim.activate(input_matrix, enable=True)
    assert np.allclose(result1, np.zeros_like(input_matrix))

    # With enable=False, should get original values
    result2 = sim.activate(input_matrix, enable=False)
    assert np.allclose(result2, input_matrix)


@pytest.mark.parametrize("dtype", [BF16, Float8, Float16])
def test_relu_passthrough(dtype):
    """Test passthrough mode (enable=False)"""
    sim = ReluSimulator(dtype)

    input_data = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    result = sim.activate(input_data, enable=False)
    assert np.allclose(result, input_data, rtol=0.01)


@pytest.mark.parametrize("dtype", [BF16, Float8, Float16])
def test_relu_edge_cases(dtype):
    """Test edge cases"""
    sim = ReluSimulator(dtype)

    # Test very small positive numbers
    small_pos = dtype.min_normal()
    result = sim.activate(np.array([small_pos]))
    assert result[0] > 0

    # Test very large numbers
    large_num = dtype.max_normal()
    result = sim.activate(np.array([large_num]))
    assert np.isclose(result[0], large_num, rtol=0.01)


@pytest.mark.parametrize("dtype", [BF16, Float8, Float16])
def test_relu_random_matrix(dtype):
    """Test with random matrix"""
    sim = ReluSimulator(dtype)

    # Generate random matrix within dtype range
    rng = np.random.default_rng(42)
    values = rng.uniform(-dtype.max_normal(), dtype.max_normal(), size=(4, 4))

    # Convert values through dtype to account for quantization
    quantized_values = np.vectorize(lambda x: dtype(x).decimal_approx)(values)

    result = sim.activate(values)
    expected = np.maximum(quantized_values, 0)

    assert np.allclose(result, expected, rtol=0.01)


def test_relu_shape_preservation():
    """Test that output maintains input shape"""
    sim = ReluSimulator()

    # Test 1D and 2D shapes
    shapes = [(10,), (3, 4), (5, 5), (2, 8)]

    for shape in shapes:
        input_data = np.random.randn(*shape)
        result = sim.activate(input_data)  # type: ignore
        assert result.shape == shape


@pytest.mark.parametrize("dtype", [BF16, Float8, Float16])
def test_relu_special_values(dtype):
    """Test special values handling"""
    sim = ReluSimulator(dtype)

    # Test zeros
    result = sim.activate(np.array([0.0, -0.0]))
    assert np.allclose(result, [0.0, 0.0])

    # Test subnormal numbers if supported by dtype
    if hasattr(dtype, "min_subnormal"):
        subnormal = dtype.min_subnormal()
        result = sim.activate(np.array([-subnormal, subnormal]))
        assert result[0] == 0
        assert result[1] > 0
