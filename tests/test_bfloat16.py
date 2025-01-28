# tests/test_bfloat16.py
import math

import pytest

from hardware_accelerators import BF16


class TestBF16Initialization:
    @pytest.mark.parametrize(
        "value,expected_binary",
        [
            (1.0, "0011111110000000"),
            (2.0, "0100000000000000"),
            (-1.0, "1011111110000000"),
            (0.0, "0000000000000000"),
            (0.5, "0011111100000000"),
            (-0.5, "1011111100000000"),
            (float("inf"), "0111111110000000"),
            (-float("inf"), "1111111110000000"),
        ],
    )
    def test_init_from_float(self, value, expected_binary):
        """Test initialization from float values"""
        bf16 = BF16(value)
        assert bf16.binary == expected_binary

    def test_init_from_binary(self):
        """Test initialization from binary strings"""
        test_cases = [
            ("0011111110000000", 1.0),
            ("0100000000000000", 2.0),
            ("1011111110000000", -1.0),
            ("0000000000000000", 0.0),
            ("0111111110000000", float("inf")),
            ("1111111110000000", -float("inf")),
        ]
        for binary, expected in test_cases:
            bf16 = BF16(binary=binary)
            if math.isinf(expected):
                assert math.isinf(float(bf16))
            else:
                assert abs(float(bf16) - expected) < 1e-3

    def test_init_invalid(self):
        """Test invalid initialization cases"""
        with pytest.raises(ValueError):
            BF16(binary="1" * 15)  # Too short
        with pytest.raises(ValueError):
            BF16(binary="1" * 17)  # Too long
        with pytest.raises(ValueError):
            BF16(binint=65536)  # Too large
        with pytest.raises(ValueError):
            BF16()  # No arguments


class TestBF16SpecialValues:
    def test_infinity(self):
        """Test infinity handling"""
        pos_inf = BF16.inf(sign=False)
        neg_inf = BF16.inf(sign=True)
        assert math.isinf(float(pos_inf))
        assert math.isinf(float(neg_inf))
        assert float(pos_inf) > 0
        assert float(neg_inf) < 0

    def test_nan(self):
        """Test NaN handling"""
        quiet_nan = BF16.nan(quiet=True)
        signaling_nan = BF16.nan(quiet=False)
        assert math.isnan(float(quiet_nan))
        assert math.isnan(float(signaling_nan))
        assert quiet_nan.binary != signaling_nan.binary

    def test_max_min_values(self):
        """Test maximum and minimum values"""
        max_val = BF16.max_value()
        min_normal = BF16.min_normal()
        min_subnormal = BF16.min_subnormal()

        assert float(max_val) == BF16.max_normal()
        assert float(min_normal) == BF16.min_normal()
        assert float(min_subnormal) == BF16.min_subnormal()


class TestBF16Arithmetic:
    @pytest.mark.parametrize(
        "a,b,expected",
        [
            (1.0, 1.0, 2.0),
            (0.5, 0.5, 1.0),
            (-1.0, 1.0, 0.0),
            (0.0, 1.0, 1.0),
            (float("inf"), 1.0, float("inf")),
            (-float("inf"), -float("inf"), -float("inf")),
        ],
    )
    def test_addition(self, a, b, expected):
        """Test addition operations"""
        bf16_a = BF16(a)
        bf16_b = BF16(b)
        result = bf16_a + bf16_b
        if math.isinf(expected):
            assert math.isinf(float(result))
        else:
            assert abs(float(result) - expected) < 1e-3

    @pytest.mark.parametrize(
        "a,b,expected",
        [
            (1.0, 1.0, 1.0),
            (0.5, 2.0, 1.0),
            (-1.0, 1.0, -1.0),
            (0.0, 1.0, 0.0),
            (float("inf"), 0.0, float("nan")),
            (float("inf"), float("inf"), float("inf")),
        ],
    )
    def test_multiplication(self, a, b, expected):
        """Test multiplication operations"""
        bf16_a = BF16(a)
        bf16_b = BF16(b)
        result = bf16_a * bf16_b
        if math.isnan(expected):
            assert math.isnan(float(result))
        elif math.isinf(expected):
            assert math.isinf(float(result))
        else:
            assert abs(float(result) - expected) < 1e-3


class TestBF16Comparisons:
    def test_comparisons(self):
        """Test all comparison operations"""
        a = BF16(1.0)
        b = BF16(2.0)
        assert a < b
        assert a <= b
        assert b > a
        assert b >= a
        assert a == a
        assert a != b

    def test_special_comparisons(self):
        """Test comparisons with special values"""
        inf = BF16.inf()
        neg_inf = BF16.inf(sign=True)
        nan = BF16.nan()
        normal = BF16(1.0)

        assert neg_inf < normal < inf
        assert not (nan == nan)  # NaN comparison rules
        assert nan != nan


class TestBF16DetailedBreakdown:
    def test_normal_breakdown(self):
        """Test detailed breakdown of normal number"""
        bf16 = BF16(1.0)
        breakdown = bf16.detailed_breakdown()
        assert breakdown["is_normal"]
        assert not breakdown["is_subnormal"]
        assert not breakdown["is_zero"]
        assert not breakdown["is_inf"]
        assert not breakdown["is_nan"]

    def test_special_value_breakdown(self):
        """Test detailed breakdown of special values"""
        inf = BF16.inf()
        nan = BF16.nan()
        zero = BF16(0.0)

        inf_breakdown = inf.detailed_breakdown()
        assert inf_breakdown["is_inf"]

        nan_breakdown = nan.detailed_breakdown()
        assert nan_breakdown["is_nan"]

        zero_breakdown = zero.detailed_breakdown()
        assert zero_breakdown["is_zero"]


class TestBF16Conversions:
    def test_float_conversion(self):
        """Test conversion to and from float32"""
        values = [1.0, -1.0, 0.5, -0.5, 0.0, float("inf"), -float("inf")]
        for val in values:
            bf16 = BF16.from_float32(val)
            if math.isinf(val):
                assert math.isinf(float(bf16))
            else:
                assert abs(float(bf16) - val) < 1e-3

    def test_string_representations(self):
        """Test string and repr conversions"""
        bf16 = BF16(1.0)
        assert str(bf16) == "1.0"
        assert repr(bf16).startswith("BF16")


class TestBF16EdgeCases:
    def test_subnormal_arithmetic(self):
        """Test arithmetic with subnormal numbers"""
        min_sub = BF16.min_subnormal()
        assert float(min_sub + min_sub) == 2 * float(min_sub)
        assert float(min_sub * BF16(2.0)) == 2 * float(min_sub)

    def test_inf_arithmetic(self):
        """Test arithmetic with infinity"""
        inf = BF16.inf()
        neg_inf = BF16.inf(sign=True)
        assert math.isinf(float(inf + inf))
        assert math.isnan(float(inf + neg_inf))
        assert math.isnan(float(inf * BF16(0.0)))
        assert math.isinf(float(inf * BF16(1.0)))

    def test_nan_propagation(self):
        """Test that NaN propagates through operations"""
        nan = BF16.nan()
        normal = BF16(1.0)
        assert math.isnan(float(nan + normal))
        assert math.isnan(float(nan * normal))
        assert math.isnan(float(nan / normal))
        assert math.isnan(float(normal / BF16(0.0)))  # Division by zero

    def test_denormal_range(self):
        """Test numbers in the denormal range"""
        min_normal = BF16.min_normal()
        max_subnormal = BF16(float(min_normal) * 0.99)
        assert float(max_subnormal) < float(min_normal)
        assert float(max_subnormal) > 0.0

    def test_gradual_underflow(self):
        """Test gradual underflow behavior"""
        x = BF16.min_normal()
        prev = float(x)
        for _ in range(5):
            x = x * BF16(0.5)
            curr = float(x)
            assert curr < prev
            prev = curr
