from .base import BaseFloat, FormatSpec


class Float8(BaseFloat):
    """
    8-bit floating point number with E4M3 format
    - 1 sign bit
    - 4 exponent bits (bias 7)
    - 3 mantissa bits
    """

    # Define format spec as class constant
    FORMAT_SPEC = FormatSpec(
        total_bits=8,
        exponent_bits=4,
        mantissa_bits=3,
        bias=7,
        # These values are pre-calculated from their binary representations
        max_normal=2**8 * 1.75,  # from 0.1111.110
        min_normal=2**-6,  # from 0.0001.000
        max_subnormal=2**-6 * (7 / 8),  # from 0.0000.111
        min_subnormal=2**-6 * (1 / 8),  # from 0.0000.001
    )

    def _get_format_spec(self) -> FormatSpec:
        return self.FORMAT_SPEC

    def _decimal_to_binary(self, num: float) -> str:
        """Convert decimal number to binary string in E4M3 format"""
        if num == 0:
            return "0.0000.000"

        # Extract sign bit
        sign = "1" if num < 0 else "0"
        num = abs(num)

        # Handle NaN
        if num != num:  # Python's way to check for NaN
            return sign + ".1111.111"

        # Clamp to max value if overflow
        if num > self.FORMAT_SPEC.max_normal:
            return "0.1111.110" if sign == "0" else "1.1111.110"

        # Find exponent and normalized mantissa
        exp = 0
        temp = num

        # Handle normal numbers
        while temp >= 2 and exp < 15:  # Changed from 8 to 15 (max exp for E4M3)
            temp /= 2
            exp += 1
        while temp < 1 and exp > -6:  # Min exp before bias is -6
            temp *= 2
            exp -= 1

        # Handle subnormal numbers
        if exp <= -6:
            # Shift mantissa right and adjust
            shift = -6 - exp
            temp /= 2**shift
            exp = -6

        # Calculate biased exponent
        if temp < 1:  # Subnormal
            biased_exp = "0000"
        else:  # Normal
            biased_exp = format(exp + self.format_spec.bias, "04b")

        # Calculate mantissa bits
        if temp < 1:  # Subnormal
            mantissa_value = int(temp * (2**self.format_spec.mantissa_bits))
        else:  # Normal
            mantissa_value = int((temp - 1) * (2**self.format_spec.mantissa_bits))

        mantissa = format(mantissa_value, f"0{self.format_spec.mantissa_bits}b")

        return f"{sign}.{biased_exp}.{mantissa}"

    def _binary_to_decimal(self, binary: str) -> float:
        """Convert binary string in E4M3 format to decimal"""
        # Clean the binary string
        binary = "".join(c for c in binary if c in "01")

        # Extract components
        sign = -1 if binary[0] == "1" else 1
        exp = binary[1:5]
        mantissa = binary[5:]

        # Handle special cases
        if exp == "1111" and mantissa == "111":
            return float("nan")
            # For all other mantissa values when exp is '1111',
            # calculate normally instead of returning max value
        if exp == "0000" and mantissa == "000":
            return 0.0

        # Convert biased exponent
        biased_exp = int(exp, 2)

        if biased_exp == 0:  # Subnormal number
            actual_exp = -6  # emin
            mantissa_value = int(mantissa, 2) / (2**self.format_spec.mantissa_bits)
            return sign * (2**actual_exp) * mantissa_value
        else:  # Normal number
            actual_exp = biased_exp - self.format_spec.bias
            mantissa_value = 1 + int(mantissa, 2) / (2**self.format_spec.mantissa_bits)
            return sign * (2**actual_exp) * mantissa_value

    @classmethod
    def from_bits(cls, bits: int) -> "Float8":
        """Create Float8 from 8-bit integer"""
        return cls(binint=bits)

    @classmethod
    def nan(cls) -> "Float8":
        """Create NaN value"""
        return cls(binary="1.1111.111")

    @classmethod
    def max_value(cls) -> "Float8":
        """Create maximum representable value"""
        return cls(binary="0.1111.110")

    @classmethod
    def min_value(cls) -> "Float8":
        """Create minimum representable normal value"""
        return cls(binary="0.0001.000")

    @classmethod
    def min_subnormal(cls) -> "Float8":
        """Create minimum representable subnormal value"""
        return cls(binary="0.0000.001")

    def detailed_breakdown(self) -> dict:
        """
        Provide detailed breakdown of the Float8 number components
        """
        binary = "".join(c for c in self.binary if c in "01")
        sign_bit = int(binary[0])
        exp_bits = binary[1:5]
        mantissa_bits = binary[5:]

        exp_val = int(exp_bits, 2)
        mantissa_val = int(mantissa_bits, 2)

        is_normal = exp_val != 0 and exp_val != 15
        is_subnormal = exp_val == 0 and mantissa_val != 0
        is_zero = exp_val == 0 and mantissa_val == 0
        is_nan = exp_val == 15 and mantissa_val == 7  # Only s.1111.111 is NaN

        return {
            "binary": self.binary,
            "sign": sign_bit,
            "exponent_bits": exp_bits,
            "exponent_value": (
                exp_val - self.format_spec.bias if exp_val != 0 else "subnormal"
            ),
            "mantissa_bits": mantissa_bits,
            "mantissa_value": mantissa_val,
            "decimal_approx": self.decimal_approx,
            "original_value": self.original_value,
            "is_normal": is_normal,
            "is_subnormal": is_subnormal,
            "is_zero": is_zero,
            "is_nan": is_nan,
            "normalized_value": (
                (1 + mantissa_val / 8) if is_normal else (mantissa_val / 8)
            ),
        }
