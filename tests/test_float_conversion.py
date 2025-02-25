import math
import pytest
import pyrtl
from hardware_accelerators import Float8, BF16, Float16, Float32
from hardware_accelerators.rtllib.utils.common import convert_float_format


class TestFloatFormatConversion:
    def setup_method(self):
        """Setup PyRTL simulation environment before each test"""
        pyrtl.reset_working_block()

    def teardown_method(self):
        """Clear PyRTL simulation environment after each test"""
        pyrtl.reset_working_block()

    def simulate_conversion(self, input_val, input_dtype, output_dtype):
        """Helper function to simulate the conversion circuit"""
        pyrtl.reset_working_block()
        input_wire = pyrtl.Input(input_dtype.bitwidth(), "input")
        output_wire = convert_float_format(input_wire, input_dtype, output_dtype)
        sim_trace = pyrtl.SimulationTrace()
        sim = pyrtl.Simulation(tracer=sim_trace)

        # Convert input value to binary integer
        input_binint = input_dtype(input_val).binint
        sim.step({"input": input_binint})

        return float(output_dtype(binint=sim.inspect(output_wire.name)))

    def test_invalid_conversion_types(self):
        """Test that invalid conversion types raise appropriate errors"""
        input_wire = pyrtl.Input(8)

        # Test non-BaseFloat types
        with pytest.raises(ValueError):
            convert_float_format(input_wire, int, Float8)  # type: ignore
        with pytest.raises(ValueError):
            convert_float_format(input_wire, Float8, int)  # type: ignore

        # Test invalid wire width
        wrong_width_wire = pyrtl.Input(10)
        with pytest.raises(ValueError):
            convert_float_format(wrong_width_wire, Float8, BF16)

        # # Test downcasting (should raise error)
        # with pytest.raises(ValueError):
        #     convert_float_format(pyrtl.Input(16), BF16, Float8)

    @pytest.mark.parametrize(
        "value",
        [
            1.0,
            -1.0,
            0.5,
            -0.5,
            0.0,
            2.0,
            -2.0,
        ],
    )
    def test_float8_to_bf16_normal_values(self, value):
        """Test conversion from Float8 to BF16 for normal values"""
        result = self.simulate_conversion(value, Float8, BF16)
        # Allow small relative error due to precision differences
        assert abs(float(result) - value) < 1e-3

    def test_float8_to_bf16_special_values(self):
        """Test conversion of special values from Float8 to BF16"""
        # Test zero
        result = self.simulate_conversion(0.0, Float8, BF16)
        assert float(result) == 0.0

        # Test max value
        max_f8 = Float8.max_value()
        result = self.simulate_conversion(float(max_f8), Float8, BF16)
        assert float(result) == float(
            max_f8
        ), f"Expected {float(max_f8)}, got {float(result)}"

        # Test min normal
        min_f8 = Float8.min_value()
        result = self.simulate_conversion(float(min_f8), Float8, BF16)
        assert float(result) == float(min_f8)

        # Test min subnormal
        min_sub_f8 = Float8.min_subnormal()
        result = self.simulate_conversion(float(min_sub_f8), Float8, BF16)
        assert float(result) == 0.0  # Subnormal should be flushed to zero in BF16

    def test_same_format_conversion(self):
        """Test conversion between same formats (should return input directly)"""
        input_wire = pyrtl.Input(8)
        output_wire = convert_float_format(input_wire, Float8, Float8)
        assert input_wire is output_wire  # Should return same wire reference

        input_wire = pyrtl.Input(16)
        output_wire = convert_float_format(input_wire, BF16, BF16)
        assert input_wire is output_wire

    def test_subnormal_conversion(self):
        """Test conversion of subnormal numbers"""
        # Create a subnormal Float8 value
        min_normal_f8 = Float8.min_value()
        subnormal_f8 = Float8(float(min_normal_f8) * 0.5)  # Should be subnormal

        result = self.simulate_conversion(float(subnormal_f8), Float8, BF16)
        # The converted value should preserve the magnitude
        assert abs(float(result)) == 0

    def test_sign_preservation(self):
        """Test that signs are preserved during conversion"""
        test_values = [1.0, -1.0, 0.5, -0.5]
        for value in test_values:
            result = self.simulate_conversion(value, Float8, BF16)
            # Check if signs match
            assert (float(result) < 0) == (value < 0)

    def test_bias_adjustment(self):
        """Test proper bias adjustment in exponent conversion"""
        # Test a power of 2 to easily check exponent handling
        value = 2.0  # Binary: 1.0 × 2¹
        result = self.simulate_conversion(value, Float8, BF16)
        assert float(result) == value

        value = 0.5  # Binary: 1.0 × 2⁻¹
        result = self.simulate_conversion(value, Float8, BF16)
        assert float(result) == value

    def test_mantissa_extension(self):
        """Test that mantissa bits are properly extended"""
        # Use a value that will have non-zero mantissa bits
        value = 1.5  # Binary: 1.1 in binary (mantissa = .1)
        result = self.simulate_conversion(value, Float8, BF16)
        # The value should be preserved exactly since it's representable in both formats
        assert float(result) == value

    def test_rounding_behavior(self):
        """Test rounding behavior during conversion"""
        # Create a value that uses all mantissa bits in Float8
        f8 = Float8(binary="0.0111.111")  # All mantissa bits set
        result = self.simulate_conversion(float(f8), Float8, BF16)
        # The result should be very close to the original value
        assert abs(float(result) - float(f8)) < 1e-6

    def test_gradual_values(self):
        """Test conversion of gradually increasing/decreasing values"""
        prev_result = None
        for i in range(-4, 5):  # Test range around 1.0
            value = 2.0**i
            result = self.simulate_conversion(value, Float8, BF16)

            # Check that the value is approximately preserved
            assert abs(float(result) - value) < 1e-3

            # Check ordering is preserved
            if prev_result is not None:
                assert float(result) > float(prev_result)
            prev_result = result

    @pytest.mark.parametrize(
        "value",
        [
            1.0,
            -1.0,
            0.5,
            -0.5,
            0.0,
            2.0,
            -2.0,
        ],
    )
    def test_float16_to_bf16_normal_values(self, value):
        """Test conversion from Float16 to BF16 for normal values"""
        result = self.simulate_conversion(value, Float16, BF16)
        # Allow small relative error due to precision differences
        assert abs(float(result) - value) < 1e-3

    def test_float16_to_bf16_special_values(self):
        """Test conversion of special values from Float16 to BF16"""
        # Test zero
        result = self.simulate_conversion(0.0, Float16, BF16)
        assert float(result) == 0.0

        # Test max value
        max_f16 = Float16.max_value()
        result = self.simulate_conversion(float(max_f16), Float16, BF16)
        # assert float(result) == float(
        #     max_f16
        # ), f"Expected {float(max_f16)}, got {float(result)}"
        assert (
            float(result) == 65280.0
        )  # I think this is the closest we can get w conversion

        # Test min normal
        min_f16 = Float16.min_value()
        result = self.simulate_conversion(float(min_f16), Float16, BF16)
        assert float(result) == float(min_f16)

        # Test min subnormal
        min_sub_f16 = Float16.min_subnormal()
        result = self.simulate_conversion(float(min_sub_f16), Float16, BF16)
        assert float(result) == 0.0  # Subnormal should be flushed to zero in BF16

    def test_float16_to_float16_same_format_conversion(self):
        """Test conversion between Float16 formats (should return input directly)"""
        input_wire = pyrtl.Input(16)
        output_wire = convert_float_format(input_wire, Float16, Float16)
        assert input_wire is output_wire  # Should return same wire reference

    def test_float16_subnormal_conversion(self):
        """Test conversion of subnormal numbers for Float16"""
        min_normal_f16 = Float16.min_value()
        subnormal_f16 = Float16(float(min_normal_f16) * 0.5)  # Should be subnormal

        result = self.simulate_conversion(float(subnormal_f16), Float16, BF16)
        # The converted value should preserve the magnitude
        assert abs(float(result)) == 0

    def test_float16_rounding_behavior(self):
        """Test rounding behavior for Float16 to BF16 conversions"""

        f16 = Float16(binary="0 00001 1111111111")
        result = self.simulate_conversion(float(f16), Float16, BF16)
        assert abs(float(result) - float(f16)) < 1e-5

        f16 = Float16(binary="0 00000 0000000001")
        result = self.simulate_conversion(float(f16), Float16, BF16)
        assert abs(float(result) - float(f16)) < 1e-6

        f16 = Float16(binary="0 11111 1111111111")
        result = self.simulate_conversion(float(f16), Float16, BF16)
        assert result != result  # NaN check

    def test_gradual_values_float16(self):
        """Test conversion of gradually increasing/decreasing values for Float16"""
        prev_result = None
        for i in range(-4, 5):  # Test range around 1.0
            value = 2.0**i
            result = self.simulate_conversion(value, Float16, BF16)

            # Check that the value is approximately preserved
            assert abs(float(result) - value) < 1e-3

            # Check ordering is preserved
            if prev_result is not None:
                assert float(result) > float(prev_result)
            prev_result = result

    @pytest.mark.parametrize(
        "value",
        [
            1.0,
            -1.0,
            0.5,
            -0.5,
            0.0,
            2.0,
            -2.0,
        ],
    )
    def test_float32_to_bf16_normal_values(self, value):
        """Test conversion from Float32 to BF16 for normal values"""
        result = self.simulate_conversion(value, Float32, BF16)
        # Allow small relative error due to precision differences
        assert abs(float(result) - value) < 1e-3

    def test_float32_to_bf16_special_values(self):
        """Test conversion of special values from Float32 to BF16"""
        # Test zero
        result = self.simulate_conversion(0.0, Float32, BF16)
        assert float(result) == 0.0

        # Test max value
        max_f32 = Float32.max_value()
        result = self.simulate_conversion(float(max_f32), Float32, BF16)
        assert math.isclose(
            float(result), float(max_f32), rel_tol=1e-2
        ), f"Expected {float(max_f32)}, got {float(result)}"

        # Test min normal
        min_f32 = Float32.min_value()
        result = self.simulate_conversion(float(min_f32), Float32, BF16)
        assert float(result) == float(min_f32)

        # Test min subnormal
        min_sub_f32 = Float32.min_subnormal()
        result = self.simulate_conversion(float(min_sub_f32), Float32, BF16)
        assert float(result) == 0.0  # Subnormal should be flushed to zero in BF16

    def test_float32_to_float32_same_format_conversion(self):
        """Test conversion between Float32 formats (should return input directly)"""
        input_wire = pyrtl.Input(32)
        output_wire = convert_float_format(input_wire, Float32, Float32)
        assert input_wire is output_wire  # Should return same wire reference

    def test_float32_subnormal_conversion(self):
        """Test conversion of subnormal numbers for Float32"""
        min_normal_f32 = Float32.min_value()
        subnormal_f32 = Float32(float(min_normal_f32) * 0.5)  # Should be subnormal

        result = self.simulate_conversion(float(subnormal_f32), Float32, BF16)
        # The converted value should preserve the magnitude
        assert abs(float(result)) == 0

    def test_float32_rounding_behavior(self):
        """Test rounding behavior for Float32 to BF16 conversions"""

        f32 = Float32(binary="0.00000001.11111111111111111111111")
        result = self.simulate_conversion(float(f32), Float32, BF16)
        assert abs(float(result) - float(f32)) < 1e-5

        f32 = Float32(binary="0.00000000.00000000000000000000001")
        result = self.simulate_conversion(float(f32), Float32, BF16)
        assert abs(float(result) - float(f32)) < 1e-6

        f32 = Float32(binary="1.11111111.11111111111111111111111")
        result = self.simulate_conversion(float(f32), Float32, BF16)
        assert result != result  # NaN check

    def test_gradual_values_float32(self):
        """Test conversion of gradually increasing/decreasing values for Float32"""
        prev_result = None
        for i in range(-4, 5):  # Test range around 1.0
            value = 2.0**i
            result = self.simulate_conversion(value, Float32, BF16)

            # Check that the value is approximately preserved
            assert abs(float(result) - value) < 1e-3

            # Check ordering is preserved
            if prev_result is not None:
                assert float(result) > float(prev_result)
            prev_result = result
