from .base import BaseFloat, FormatSpec


class Float32(BaseFloat):
    """
    32-bit floating point number with IEEE 754 single-precision format
    - 1 sign bit
    - 8 exponent bits (bias 127)
    - 23 mantissa bits
    """

    @classmethod
    def format_spec(cls) -> FormatSpec:
        return FormatSpec(
            total_bits=32,
            exponent_bits=8,
            mantissa_bits=23,
            bias=127,
            max_normal=3.4028235e38,  # from 0.11111110.11111111111111111111111
            min_normal=2**-126,  # from 0.00000001.00000000000000000000000
            max_subnormal=2**-126
            * (8388607 / 8388608),  # from 0.00000000.11111111111111111111111
            min_subnormal=2**-149,  # from 0.00000000.00000000000000000000001
        )

    @classmethod
    def binary_max(cls) -> int:
        return 0b01111111011111111111111111111111

    def _decimal_to_binary(self, num: float) -> str:
        """Convert decimal number to binary string in IEEE 754 format"""
        if num == 0:
            return "0.00000000.00000000000000000000000"

        # Extract sign bit
        sign = "1" if num < 0 else "0"
        num = abs(num)

        # Handle NaN
        if num != num:  # Python's way to check for NaN
            return sign + ".11111111.11111111111111111111111"

        # Clamp to max value if overflow
        if num > self.max_normal():
            return (
                "0.11111110.11111111111111111111111"
                if sign == "0"
                else "1.11111110.11111111111111111111111"
            )

        # Find exponent and normalized mantissa
        exp = 0
        temp = num

        # Handle normal numbers
        while temp >= 2 and exp < 255:
            temp /= 2
            exp += 1
        while temp < 1 and exp > -126:
            temp *= 2
            exp -= 1

        # Handle subnormal numbers
        if exp <= -126:
            # Shift mantissa right and adjust
            shift = -126 - exp
            temp /= 2**shift
            exp = -126

        # Calculate biased exponent
        if temp < 1:  # Subnormal
            biased_exp = "00000000"
        else:  # Normal
            biased_exp = format(exp + self.bias(), "08b")

        # Calculate mantissa bits
        if temp < 1:  # Subnormal
            mantissa_value = int(temp * (2 ** self.mantissa_bits()))
        else:  # Normal
            mantissa_value = int((temp - 1) * (2 ** self.mantissa_bits()))

        mantissa = format(mantissa_value, f"0{self.mantissa_bits()}b")

        return f"{sign}.{biased_exp}.{mantissa}"

    def _binary_to_decimal(self, binary: str) -> float:
        """Convert binary string in IEEE 754 format to decimal"""
        # Clean up binary string
        binary = "".join(c for c in binary if c in "01")

        # Extract components
        sign = -1 if binary[0] == "1" else 1
        exp = binary[1:9]
        mantissa = binary[9:]

        # Handle special cases
        if (
            exp == "11111111" and mantissa == "11111111111111111111111"
        ):  # NaN representation
            return float("nan")
        if exp == "00000000" and mantissa == "00000000000000000000000":
            return 0.0

        # Convert biased exponent
        biased_exp = int(exp, 2)

        if biased_exp == 0:  # Subnormal number
            actual_exp = -126
            mantissa_value = int(mantissa, 2) / (2 ** self.mantissa_bits())
            return sign * (2**actual_exp) * mantissa_value
        else:  # Normal number
            actual_exp = biased_exp - self.bias()
            mantissa_value = 1 + int(mantissa, 2) / (2 ** self.mantissa_bits())
            return sign * (2**actual_exp) * mantissa_value

    @classmethod
    def from_bits(cls, bits: int) -> "Float32":
        """Create Float32 from 32-bit integer"""
        return cls(binint=bits)

    @classmethod
    def nan(cls) -> "Float32":
        """Create NaN value"""
        return cls(binary="1.11111111.11111111111111111111111")

    @classmethod
    def max_value(cls) -> "Float32":
        """Create maximum representable value"""
        return cls(binary="0.11111110.11111111111111111111111")

    @classmethod
    def min_value(cls) -> "Float32":
        """Create minimum representable normal value"""
        return cls(binary="0.00000001.00000000000000000000000")

    @classmethod
    def min_subnormal(cls) -> "Float32":
        """Create minimum representable subnormal value"""
        return cls(binary="0.00000000.00000000000000000000001")

    def detailed_breakdown(self) -> dict:
        """Provide detailed breakdown of the Float32 number components"""
        binary = "".join(c for c in self.binary if c in "01")
        sign_bit = int(binary[0])
        exp_bits = binary[1:9]
        mantissa_bits = binary[9:]

        exp_val = int(exp_bits, 2)
        mantissa_val = int(mantissa_bits, 2)

        is_normal = exp_val != 0 and exp_val != 255
        is_subnormal = exp_val == 0 and mantissa_val != 0
        is_zero = exp_val == 0 and mantissa_val == 0
        is_nan = (
            exp_val == 255 and mantissa_val == 8388607
        )  # Only s.11111111.11111111111111111111111 is NaN

        return {
            "binary": self.binary,
            "sign": sign_bit,
            "exponent_bits": exp_bits,
            "exponent_value": (exp_val - self.bias() if exp_val != 0 else "subnormal"),
            "mantissa_bits": mantissa_bits,
            "mantissa_value": mantissa_val,
            "decimal_approx": self.decimal_approx,
            "original_value": self.original_value,
            "is_normal": is_normal,
            "is_subnormal": is_subnormal,
            "is_zero": is_zero,
            "is_nan": is_nan,
            "normalized_value": (
                (1 + mantissa_val / 8388608) if is_normal else (mantissa_val / 8388608)
            ),
        }
