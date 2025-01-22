import pytest
import numpy as np
from hardware_accelerators.dtypes import Float8, BF16
from hardware_accelerators.dtypes.conversions import convert_float_type, quantize_weights

class TestFloatConversions:
    def test_bf16_to_float8(self):
        """Test conversion from BF16 to Float8"""
        # Test normal numbers
        bf16_val = BF16(1.5)
        fp8_val = convert_float_type(bf16_val, Float8)
        assert abs(float(fp8_val) - 1.5) < 0.1
        
        # Test small numbers
        bf16_val = BF16(0.125)
        fp8_val = convert_float_type(bf16_val, Float8)
        assert abs(float(fp8_val) - 0.125) < 0.01
        
        # Test signs
        bf16_val = BF16(-2.0)
        fp8_val = convert_float_type(bf16_val, Float8)
        assert float(fp8_val) < 0
        assert abs(float(fp8_val) + 2.0) < 0.1
        
    def test_float8_to_bf16(self):
        """Test conversion from Float8 to BF16"""
        # Test normal numbers
        fp8_val = Float8(1.5)
        bf16_val = convert_float_type(fp8_val, BF16)
        assert abs(float(bf16_val) - 1.5) < 0.01
        
        # Test small numbers
        fp8_val = Float8(0.125)
        bf16_val = convert_float_type(fp8_val, BF16)
        assert abs(float(bf16_val) - 0.125) < 0.001
        
        # Test signs
        fp8_val = Float8(-2.0)
        bf16_val = convert_float_type(fp8_val, BF16)
        assert float(bf16_val) < 0
        assert abs(float(bf16_val) + 2.0) < 0.01

class TestWeightQuantization:
    def test_simple_quantization(self):
        """Test basic weight quantization without scaling"""
        weights = np.array([1.5, -2.0, 0.5, -0.25])
        quantized = quantize_weights(weights, Float8)
        assert isinstance(quantized, np.ndarray)
        assert quantized.shape == weights.shape
        assert abs(quantized[0] - 1.5) < 0.1
        assert abs(quantized[1] + 2.0) < 0.1
        
    def test_scaled_quantization(self):
        """Test weight quantization with scaling"""
        weights = np.array([10.0, -15.0, 5.0, -2.5])
        result = quantize_weights(weights, Float8, scale_factor=10.0)
        assert isinstance(result, dict)
        assert 'weights' in result
        assert 'scale' in result
        assert result['scale'] == 10.0
        
        # Check scaled values
        quantized = result['weights']
        assert abs(quantized[0] * 10.0 - 10.0) < 1.0
        assert abs(quantized[1] * 10.0 + 15.0) < 1.0
        
    def test_per_channel_quantization(self):
        """Test per-channel weight quantization"""
        # Create 2D weight matrix (2 output channels, 3 input features)
        weights = np.array([
            [1.0, 2.0, 3.0],  # Channel 1
            [10.0, 20.0, 30.0]  # Channel 2
        ])
        
        result = quantize_weights(weights, Float8, per_channel=True)
        assert isinstance(result, dict)
        assert 'weights' in result
        assert 'scales' in result
        assert len(result['scales']) == 2  # One scale per channel
        
        # Check that larger channel has larger scale
        assert result['scales'][1] > result['scales'][0]
        
        # Check values are properly scaled
        quantized = result['weights']
        assert abs(quantized[0, 0] * result['scales'][0] - 1.0) < 0.1
        assert abs(quantized[1, 0] * result['scales'][1] - 10.0) < 1.0 