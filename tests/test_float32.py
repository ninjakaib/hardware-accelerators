# tests/test_float32.py
import math

import pytest

from hardware_accelerators import Float32


class TestFloat32Initialization:
    def test_init_from_float(self, sample_float32_values):
        """Test initialization from float values"""
        for value, expected_binary in sample_float32_values:
            f32 = Float32(value)
            assert f32.binary == expected_binary

    def test_init_from_binary(self):
        """Test initialization from binary strings"""
        test_cases = [
            ("00111111000000000000000000000000", 0.5),
            ("00111111100000000000000000000000", 1.0),
            ("01000000000000000000000000000000", 2.0),
            ("10111111100000000000000000000000", -1.0),
            ("00000000000000000000000000000000", 0.0),
            ("01111111011111111111111111111111", Float32.max_value().decimal_approx),
            (
                "00000000000000000000000000000001",
                Float32.min_subnormal().decimal_approx,
            ),
        ]
        for binary, expected in test_cases:
            f32 = Float32(binary)
            assert abs(f32.decimal_approx - expected) < 1e-6

    def test_init_from_binint(self):
        """Test initialization from binary integers"""
        test_cases = [
            (0b00111111000000000000000000000000, 0.5),
            (0b00111111100000000000000000000000, 1.0),
            (0b01000000000000000000000000000000, 2.0),
            (0b10111111100000000000000000000000, -1.0),
            (0b00000000000000000000000000000000, 0.0),
        ]
        for binint, expected in test_cases:
            f32 = Float32(binint=binint)
            assert abs(f32.decimal_approx - expected) < 1e-6

    def test_invalid_initialization(self):
        """Test invalid initialization cases"""
        with pytest.raises(ValueError):
            Float32(binary="0111111101111111111111111111111")  # Too short
        with pytest.raises(ValueError):
            Float32(binary="011111110111111111111111111111111")  # Too long
        with pytest.raises(ValueError):
            Float32(binint=2**32)  # Too large
        with pytest.raises(ValueError):
            Float32(binint=-1)  # Negative
        with pytest.raises(ValueError):
            Float32()  # No arguments


class TestFloat32SpecialValues:
    def test_nan(self):
        """Test NaN handling"""
        nan = Float32.nan()
        assert math.isnan(float(nan))
        assert nan.binary == "1.11111111.11111111111111111111111"

    def test_max_value(self):
        """Test maximum normal value"""
        max_val = Float32.max_value()
        assert max_val.binary == "0.11111110.11111111111111111111111"
        assert math.isclose(float(max_val), Float32.max_normal(), rel_tol=1e-8)

    def test_min_value(self):
        """Test minimum normal value"""
        min_val = Float32.min_value()
        assert min_val.binary == "0.00000001.00000000000000000000000"
        assert float(min_val) == Float32.min_normal()

    def test_min_subnormal(self):
        """Test minimum subnormal value"""
        min_sub = Float32.min_subnormal()
        assert min_sub.binary == "0.00000000.00000000000000000000001"
        assert float(min_sub) == Float32.min_subnormal()


class TestFloat32Arithmetic:
    @pytest.mark.parametrize(
        "a,b,expected",
        [
            (1.0, 1.0, 2.0),
            (0.5, 0.5, 1.0),
            (-1.0, 1.0, 0.0),
            (1.5, 2.5, 4.0),
        ],
    )
    def test_addition(self, a, b, expected):
        """Test addition operations"""
        f32_a = Float32(a)
        f32_b = Float32(b)
        result = f32_a + f32_b
        assert abs(float(result) - expected) < 1e-6
        # Test reverse addition
        result = f32_a + b
        assert abs(float(result) - expected) < 1e-6

    @pytest.mark.parametrize(
        "a,b,expected",
        [
            (2.0, 3.0, 6.0),
            (1.5, 4.0, 6.0),
            (-1.0, 2.0, -2.0),
            (0.5, 0.5, 0.25),
        ],
    )
    def test_multiplication(self, a, b, expected):
        """Test multiplication operations"""
        f32_a = Float32(a)
        f32_b = Float32(b)
        result = f32_a * f32_b
        assert abs(float(result) - expected) < 1e-6
        # Test reverse multiplication
        result = f32_a * b
        assert abs(float(result) - expected) < 1e-6

    @pytest.mark.parametrize(
        "a,b,expected",
        [
            (2.0, 1.0, 1.0),
            (1.0, 0.5, 0.5),
            (0.0, 1.0, -1.0),
            (-1.0, -1.0, 0.0),
        ],
    )
    def test_subtraction(self, a, b, expected):
        """Test subtraction operations"""
        f32_a = Float32(a)
        f32_b = Float32(b)
        result = f32_a - f32_b
        assert abs(float(result) - expected) < 1e-6
        # Test reverse subtraction
        result = a - f32_b
        assert abs(float(result) - expected) < 1e-6

    @pytest.mark.parametrize(
        "a,b,expected",
        [
            (1.0, 1.0, 1.0),
            (1.0, 2.0, 0.5),
            (-1.0, 2.0, -0.5),
            (0.0, 1.0, 0.0),
        ],
    )
    def test_division(self, a, b, expected):
        """Test division operations"""
        f32_a = Float32(a)
        f32_b = Float32(b)
        result = f32_a / f32_b
        assert abs(float(result) - expected) < 1e-6
        # Test reverse division
        result = a / f32_b
        assert abs(float(result) - expected) < 1e-6

    def test_arithmetic_with_zero(self):
        """Test arithmetic operations with zero"""
        f32 = Float32(1.0)
        zero = Float32(0.0)

        assert float(f32 + zero) == 1.0
        assert float(f32 + zero) - 1.0 == 0.0
        assert float(f32 * zero) == 0.0
        assert float(f32 - zero) == 1.0
        result = f32 / zero
        assert math.isnan(float(result))


class TestFloat32Comparisons:
    def test_equality(self):
        """Test equality comparisons"""
        assert Float32(1.0) == Float32(1.0)
        assert Float32(1.0) == 1.0
        assert not (Float32(1.0) == Float32(2.0))

    def test_less_than(self):
        """Test less than comparisons"""
        assert Float32(1.0) < Float32(2.0)
        assert Float32(1.0) < Float32(1.1)
        assert Float32(1.0) < 1.1
        assert not (Float32(2.0) < Float32(1.9))

    def test_greater_than(self):
        """Test greater than comparisons"""
        assert Float32(2.0) > Float32(1.0)
        assert Float32(1.1) > Float32(1.0)
        assert Float32(1.1) > 1.0
        assert not (Float32(1.0) > Float32(1.9))


class TestFloat32Conversions:
    def test_to_float(self):
        """Test conversion to float"""
        f32 = Float32(1.0)
        assert float(f32) == 1.0

    def test_str_representation(self):
        """Test string representation"""
        f32 = Float32(1.0)
        assert str(f32) == "1.0"
        assert repr(f32).startswith("Float32")


class TestFloat32DetailedBreakdown:
    def test_normal_number_breakdown(self):
        """Test detailed breakdown of normal number"""
        f32 = Float32(1.0)
        breakdown = f32.detailed_breakdown()
        assert breakdown["is_normal"] == True
        assert breakdown["is_subnormal"] == False
        assert breakdown["is_zero"] == False
        assert breakdown["is_nan"] == False

    def test_subnormal_number_breakdown(self):
        """Test detailed breakdown of subnormal number"""
        f32 = Float32.min_subnormal()
        breakdown = f32.detailed_breakdown()
        assert breakdown["is_normal"] == False
        assert breakdown["is_subnormal"] == True
        assert breakdown["is_zero"] == False
        assert breakdown["is_nan"] == False

    def test_zero_breakdown(self):
        """Test detailed breakdown of zero"""
        f32 = Float32(0.0)
        breakdown = f32.detailed_breakdown()
        assert breakdown["is_normal"] == False
        assert breakdown["is_subnormal"] == False
        assert breakdown["is_zero"] == True
        assert breakdown["is_nan"] == False

    def test_nan_breakdown(self):
        """Test detailed breakdown of NaN"""
        f32 = Float32.nan()
        breakdown = f32.detailed_breakdown()
        assert breakdown["is_normal"] == False
        assert breakdown["is_subnormal"] == False
        assert breakdown["is_zero"] == False
        assert breakdown["is_nan"] == True


class TestFloat32EdgeCases:
    def test_subnormal_arithmetic(self):
        """Test arithmetic with subnormal numbers"""
        min_sub = Float32.min_subnormal()
        assert float(min_sub + min_sub) == 2 * float(min_sub)
        assert float(min_sub * Float32(2.0)) == 2 * float(min_sub)

    def test_overflow_handling(self):
        """Test handling of overflow cases"""
        max_val = Float32.max_value()
        result = max_val * Float32(2.0)
        # Test that result is either infinity or max_value
        assert math.isinf(float(result)) or float(result) == float(max_val)

    def test_gradual_overflow(self):
        """Test behavior approaching overflow"""
        large_val = Float32(1.7014117e38)  # Half of max_value
        doubled = large_val * Float32(2.0)
        assert float(doubled) <= float(Float32.max_value())

    def test_underflow_handling(self):
        """Test handling of underflow cases"""
        min_sub = Float32.min_subnormal()
        result = min_sub * Float32(0.5)
        assert float(result) == 0.0  # Should underflow to zero

    def test_denormal_range(self):
        """Test numbers in the denormal range"""
        min_normal = Float32.min_value()
        max_subnormal = Float32(float(min_normal) * 0.99)
        assert float(max_subnormal) < float(min_normal)
        assert float(max_subnormal) > 0.0

    def test_max_value_operations(self):
        """Test operations with maximum values (Float32 doesn't support infinity)"""
        max_val = Float32.max_value()
        assert float(max_val * Float32(1.0)) == float(max_val)
        assert float(max_val + Float32(1.0)) == float(max_val)  # Should clamp
        assert float(max_val - Float32(max_val)) == 0.0
        assert float(max_val * Float32(0.0)) == 0.0
