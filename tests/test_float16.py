# tests/test_float16.py
import math

import pytest

from hardware_accelerators import Float16


class TestFloat16Initialization:
    def test_init_from_float(self, sample_float16_values):
        """Test initialization from float values"""
        for value, expected_binary in sample_float16_values:
            f16 = Float16(value)
            assert f16.binary == expected_binary

    def test_init_from_binary(self):
        """Test initialization from binary strings"""
        test_cases = [
            ("0011100000000000", 0.5),
            ("0011110000000000", 1.0),
            ("0100000000000000", 2.0),
            ("1011110000000000", -1.0),
            ("0000000000000000", 0.0),
            ("0111101111111111", Float16.max_value().decimal_approx),
            ("0000000000000001", Float16.min_subnormal().decimal_approx),
        ]
        for binary, expected in test_cases:
            f16 = Float16(binary)
            assert abs(f16.decimal_approx - expected) < 1e-6

    def test_init_from_binint(self):
        """Test initialization from binary integers"""
        test_cases = [
            (0b0011100000000000, 0.5),
            (0b0011110000000000, 1.0),
            (0b0100000000000000, 2.0),
            (0b1011110000000000, -1.0),
            (0b0000000000000000, 0.0),
        ]
        for binint, expected in test_cases:
            f16 = Float16(binint=binint)
            assert abs(f16.decimal_approx - expected) < 1e-6

    def test_invalid_initialization(self):
        """Test invalid initialization cases"""
        with pytest.raises(ValueError):
            Float16(binary="011110111111111")  # Too short
        with pytest.raises(ValueError):
            Float16(binary="01111011111111111")  # Too long
        with pytest.raises(ValueError):
            Float16(binint=65536)  # Too large
        with pytest.raises(ValueError):
            Float16(binint=-1)  # Negative
        with pytest.raises(ValueError):
            Float16()  # No arguments


class TestFloat16SpecialValues:
    def test_nan(self):
        """Test NaN handling"""
        nan = Float16.nan()
        assert math.isnan(float(nan))
        assert nan.binary == "1.11111.1111111111"

    def test_max_value(self):
        """Test maximum normal value"""
        max_val = Float16.max_value()
        assert max_val.binary == "0.11110.1111111111"
        assert float(max_val) == Float16.max_normal()

    def test_min_value(self):
        """Test minimum normal value"""
        min_val = Float16.min_value()
        assert min_val.binary == "0.00001.0000000000"
        assert float(min_val) == Float16.min_normal()

    def test_min_subnormal(self):
        """Test minimum subnormal value"""
        min_sub = Float16.min_subnormal()
        assert min_sub.binary == "0.00000.0000000001"
        assert float(min_sub) == Float16.min_subnormal()


class TestFloat16Arithmetic:
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
        f16_a = Float16(a)
        f16_b = Float16(b)
        result = f16_a + f16_b
        assert abs(float(result) - expected) < 1e-6
        # Test reverse addition
        result = f16_a + b
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
        f16_a = Float16(a)
        f16_b = Float16(b)
        result = f16_a * f16_b
        assert abs(float(result) - expected) < 1e-6
        # Test reverse multiplication
        result = f16_a * b
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
        f16_a = Float16(a)
        f16_b = Float16(b)
        result = f16_a - f16_b
        assert abs(float(result) - expected) < 1e-6
        # Test reverse subtraction
        result = a - f16_b
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
        f16_a = Float16(a)
        f16_b = Float16(b)
        result = f16_a / f16_b
        assert abs(float(result) - expected) < 1e-6
        # Test reverse division
        result = a / f16_b
        assert abs(float(result) - expected) < 1e-6

    def test_arithmetic_with_zero(self):
        """Test arithmetic operations with zero"""
        f16 = Float16(1.0)
        zero = Float16(0.0)

        assert float(f16 + zero) == 1.0
        assert float(f16 + zero) - 1.0 == 0.0
        assert float(f16 * zero) == 0.0
        assert float(f16 - zero) == 1.0
        result = f16 / zero
        assert math.isnan(float(result))


class TestFloat16Comparisons:
    def test_equality(self):
        """Test equality comparisons"""
        assert Float16(1.0) == Float16(1.0)
        assert Float16(1.0) == 1.0
        assert not (Float16(1.0) == Float16(2.0))

    def test_less_than(self):
        """Test less than comparisons"""
        assert Float16(1.0) < Float16(2.0)
        assert Float16(1.0) < Float16(1.1)
        assert Float16(1.0) < 1.1
        assert not (Float16(2.0) < Float16(1.9))

    def test_greater_than(self):
        """Test greater than comparisons"""
        assert Float16(2.0) > Float16(1.0)
        assert Float16(1.1) > Float16(1.0)
        assert Float16(1.1) > 1.0
        assert not (Float16(1.0) > Float16(1.9))


class TestFloat16Conversions:
    def test_to_float(self):
        """Test conversion to float"""
        f16 = Float16(1.0)
        assert float(f16) == 1.0

    def test_str_representation(self):
        """Test string representation"""
        f16 = Float16(1.0)
        assert str(f16) == "1.0"
        assert repr(f16).startswith("Float16")


class TestFloat16DetailedBreakdown:
    def test_normal_number_breakdown(self):
        """Test detailed breakdown of normal number"""
        f16 = Float16(1.0)
        breakdown = f16.detailed_breakdown()
        assert breakdown["is_normal"] == True
        assert breakdown["is_subnormal"] == False
        assert breakdown["is_zero"] == False
        assert breakdown["is_nan"] == False

    def test_subnormal_number_breakdown(self):
        """Test detailed breakdown of subnormal number"""
        f16 = Float16.min_subnormal()
        breakdown = f16.detailed_breakdown()
        assert breakdown["is_normal"] == False
        assert breakdown["is_subnormal"] == True
        assert breakdown["is_zero"] == False
        assert breakdown["is_nan"] == False

    def test_zero_breakdown(self):
        """Test detailed breakdown of zero"""
        f16 = Float16(0.0)
        breakdown = f16.detailed_breakdown()
        assert breakdown["is_normal"] == False
        assert breakdown["is_subnormal"] == False
        assert breakdown["is_zero"] == True
        assert breakdown["is_nan"] == False

    def test_nan_breakdown(self):
        """Test detailed breakdown of NaN"""
        f16 = Float16.nan()
        breakdown = f16.detailed_breakdown()
        assert breakdown["is_normal"] == False
        assert breakdown["is_subnormal"] == False
        assert breakdown["is_zero"] == False
        assert breakdown["is_nan"] == True


class TestFloat16EdgeCases:
    def test_subnormal_arithmetic(self):
        """Test arithmetic with subnormal numbers"""
        min_sub = Float16.min_subnormal()
        assert float(min_sub + min_sub) == 2 * float(min_sub)
        assert float(min_sub * Float16(2.0)) == 2 * float(min_sub)

    def test_overflow_handling(self):
        """Test handling of overflow cases"""
        max_val = Float16.max_value()
        result = max_val * Float16(2.0)
        # Test that result is either infinity or max_value
        assert math.isinf(float(result)) or float(result) == float(max_val)

    def test_gradual_overflow(self):
        """Test behavior approaching overflow"""
        large_val = Float16(224.0)  # Half of max_value
        doubled = large_val * Float16(2.0)
        assert float(doubled) <= float(Float16.max_value())

    def test_underflow_handling(self):
        """Test handling of underflow cases"""
        min_sub = Float16.min_subnormal()
        result = min_sub * Float16(0.5)
        assert float(result) == 0.0  # Should underflow to zero

    def test_denormal_range(self):
        """Test numbers in the denormal range"""
        min_normal = Float16.min_value()
        max_subnormal = Float16(float(min_normal) * 0.99)
        assert float(max_subnormal) < float(min_normal)
        assert float(max_subnormal) > 0.0

    def test_max_value_operations(self):
        """Test operations with maximum values (Float16 doesn't support infinity)"""
        max_val = Float16.max_value()
        assert float(max_val * Float16(1.0)) == float(max_val)
        assert float(max_val + Float16(1.0)) == float(max_val)  # Should clamp
        assert float(max_val - Float16(max_val)) == 0.0
        assert float(max_val * Float16(0.0)) == 0.0
