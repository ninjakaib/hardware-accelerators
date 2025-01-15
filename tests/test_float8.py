# tests/test_float8.py
import pytest
import math
from hardware_accelerators import Float8


class TestFloat8Initialization:
    def test_init_from_float(self, sample_float8_values):
        """Test initialization from float values"""
        for value, expected_binary in sample_float8_values:
            f8 = Float8(value)
            assert f8.binary == expected_binary

    def test_init_from_binary(self):
        """Test initialization from binary strings"""
        test_cases = [
            ("00111000", 1.0),
            ("00110000", 0.5),
            ("10111000", -1.0),
            ("00000000", 0.0),
            ("01111110", Float8.max_value().decimal_approx),
            ("00000001", Float8.min_subnormal().decimal_approx),
        ]
        for binary, expected in test_cases:
            f8 = Float8(binary)
            assert abs(f8.decimal_approx - expected) < 1e-6

    def test_init_from_binint(self):
        """Test initialization from binary integers"""
        test_cases = [
            (0b00111000, 1.0),
            (0b00110000, 0.5),
            (0b10111000, -1.0),
            (0b00000000, 0.0),
        ]
        for binint, expected in test_cases:
            f8 = Float8(binint=binint)
            assert abs(f8.decimal_approx - expected) < 1e-6

    def test_invalid_initialization(self):
        """Test invalid initialization cases"""
        with pytest.raises(ValueError):
            Float8(binary="1111111")  # Too short
        with pytest.raises(ValueError):
            Float8(binary="111111111")  # Too long
        with pytest.raises(ValueError):
            Float8(binint=256)  # Too large
        with pytest.raises(ValueError):
            Float8(binint=-1)  # Negative
        with pytest.raises(ValueError):
            Float8()  # No arguments


class TestFloat8SpecialValues:
    def test_nan(self):
        """Test NaN handling"""
        nan = Float8.nan()
        assert math.isnan(float(nan))
        assert nan.binary == "1.1111.111"

    def test_max_value(self):
        """Test maximum value"""
        max_val = Float8.max_value()
        assert max_val.binary == "0.1111.110"
        assert float(max_val) == Float8.FORMAT_SPEC.max_normal

    def test_min_value(self):
        """Test minimum normal value"""
        min_val = Float8.min_value()
        assert min_val.binary == "0.0001.000"
        assert float(min_val) == Float8.FORMAT_SPEC.min_normal

    def test_min_subnormal(self):
        """Test minimum subnormal value"""
        min_sub = Float8.min_subnormal()
        assert min_sub.binary == "0.0000.001"
        assert float(min_sub) == Float8.FORMAT_SPEC.min_subnormal


class TestFloat8Arithmetic:
    @pytest.mark.parametrize(
        "a,b,expected",
        [
            (1.0, 1.0, 2.0),
            (0.5, 0.5, 1.0),
            (-1.0, 1.0, 0.0),
            (0.0, 1.0, 1.0),
        ],
    )
    def test_addition(self, a, b, expected):
        """Test addition operations"""
        f8_a = Float8(a)
        f8_b = Float8(b)
        result = f8_a + f8_b
        assert abs(float(result) - expected) < 1e-6
        # Test reverse addition
        result = f8_a + b
        assert abs(float(result) - expected) < 1e-6

    @pytest.mark.parametrize(
        "a,b,expected",
        [
            (1.0, 1.0, 1.0),
            (0.5, 2.0, 1.0),
            (-1.0, 1.0, -1.0),
            (0.0, 1.0, 0.0),
        ],
    )
    def test_multiplication(self, a, b, expected):
        """Test multiplication operations"""
        f8_a = Float8(a)
        f8_b = Float8(b)
        result = f8_a * f8_b
        assert abs(float(result) - expected) < 1e-6
        # Test reverse multiplication
        result = f8_a * b
        assert abs(float(result) - expected) < 1e-6

    # Add subtraction and division tests
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
        f8_a = Float8(a)
        f8_b = Float8(b)
        result = f8_a - f8_b
        assert abs(float(result) - expected) < 1e-6
        # Test reverse subtraction
        result = a - f8_b
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
        f8_a = Float8(a)
        f8_b = Float8(b)
        result = f8_a / f8_b
        assert abs(float(result) - expected) < 1e-6
        # Test reverse division
        result = a / f8_b
        assert abs(float(result) - expected) < 1e-6

    def test_arithmetic_with_zero(self):
        """Test arithmetic operations with zero"""
        f8 = Float8(1.0)
        zero = Float8(0.0)
        assert float(f8 + zero) == 1.0
        assert float(f8 * zero) == 0.0
        assert float(f8 - zero) == 1.0
        result = f8 / zero
        assert math.isnan(float(result))


class TestFloat8Comparisons:
    def test_equality(self):
        """Test equality comparisons"""
        assert Float8(1.0) == Float8(1.0)
        assert Float8(1.0) == 1.0
        assert not (Float8(1.0) == Float8(2.0))

    def test_less_than(self):
        """Test less than comparisons"""
        assert Float8(1.0) < Float8(2.0)
        assert Float8(1.0) < 2.0
        assert not (Float8(2.0) < Float8(1.0))

    def test_greater_than(self):
        """Test greater than comparisons"""
        assert Float8(2.0) > Float8(1.0)
        assert Float8(2.0) > 1.0
        assert not (Float8(1.0) > Float8(2.0))


class TestFloat8Conversions:
    def test_to_float(self):
        """Test conversion to float"""
        f8 = Float8(1.0)
        assert float(f8) == 1.0

    def test_str_representation(self):
        """Test string representation"""
        f8 = Float8(1.0)
        assert str(f8) == "1.0"
        assert repr(f8).startswith("Float8")


class TestFloat8DetailedBreakdown:
    def test_normal_number_breakdown(self):
        """Test detailed breakdown of normal number"""
        f8 = Float8(1.0)
        breakdown = f8.detailed_breakdown()
        assert breakdown["is_normal"] == True
        assert breakdown["is_subnormal"] == False
        assert breakdown["is_zero"] == False
        assert breakdown["is_nan"] == False

    def test_subnormal_number_breakdown(self):
        """Test detailed breakdown of subnormal number"""
        f8 = Float8.min_subnormal()
        breakdown = f8.detailed_breakdown()
        assert breakdown["is_normal"] == False
        assert breakdown["is_subnormal"] == True
        assert breakdown["is_zero"] == False
        assert breakdown["is_nan"] == False

    def test_zero_breakdown(self):
        """Test detailed breakdown of zero"""
        f8 = Float8(0.0)
        breakdown = f8.detailed_breakdown()
        assert breakdown["is_normal"] == False
        assert breakdown["is_subnormal"] == False
        assert breakdown["is_zero"] == True
        assert breakdown["is_nan"] == False

    def test_nan_breakdown(self):
        """Test detailed breakdown of NaN"""
        f8 = Float8.nan()
        breakdown = f8.detailed_breakdown()
        assert breakdown["is_normal"] == False
        assert breakdown["is_subnormal"] == False
        assert breakdown["is_zero"] == False
        assert breakdown["is_nan"] == True


class TestFloat8EdgeCases:
    def test_subnormal_arithmetic(self):
        """Test arithmetic with subnormal numbers"""
        min_sub = Float8.min_subnormal()
        assert float(min_sub + min_sub) == 2 * float(min_sub)
        assert float(min_sub * Float8(2.0)) == 2 * float(min_sub)

    def test_overflow_handling(self):
        """Test handling of overflow cases"""
        max_val = Float8.max_value()
        result = max_val * Float8(2.0)
        # Test that result is either infinity or max_value
        assert math.isinf(float(result)) or float(result) == float(max_val)

    def test_gradual_overflow(self):
        """Test behavior approaching overflow"""
        large_val = Float8(224.0)  # Half of max_value
        doubled = large_val * Float8(2.0)
        assert float(doubled) <= float(Float8.max_value())

    def test_underflow_handling(self):
        """Test handling of underflow cases"""
        min_sub = Float8.min_subnormal()
        result = min_sub * Float8(0.5)
        assert float(result) == 0.0  # Should underflow to zero

    def test_denormal_range(self):
        """Test numbers in the denormal range"""
        min_normal = Float8.min_value()
        max_subnormal = Float8(float(min_normal) * 0.99)
        assert float(max_subnormal) < float(min_normal)
        assert float(max_subnormal) > 0.0

    def test_max_value_operations(self):
        """Test operations with maximum values (Float8 doesn't support infinity)"""
        max_val = Float8.max_value()
        assert float(max_val * Float8(1.0)) == float(max_val)
        assert float(max_val + Float8(1.0)) == float(max_val)  # Should clamp
        assert float(max_val - Float8(max_val)) == 0.0
        assert float(max_val * Float8(0.0)) == 0.0
