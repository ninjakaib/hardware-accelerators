import struct
from typing import Tuple
from .base import BaseFloat, FormatSpec


class BF16(BaseFloat):
    """
    16-bit Brain Floating Point (bfloat16)
    - 1 sign bit
    - 8 exponent bits (bias 127)
    - 7 mantissa bits
    """

    FORMAT_SPEC = FormatSpec(
        total_bits=16,
        exponent_bits=8,
        mantissa_bits=7,
        bias=127,
        # max_normal: exp=11111110 (254-127=127), mantissa=1111111 (1 + 127/128)
        max_normal=2**127 * (1 + 127 / 128),
        # min_normal: exp=00000001 (1-127=-126), mantissa=0000000 (1 + 0/128)
        min_normal=2**-126 * (1 + 0 / 128),
        # max_subnormal: exp=00000000 (-126), mantissa=1111111 (127/128)
        max_subnormal=2**-126 * (127 / 128),
        # min_subnormal: exp=00000000 (-126), mantissa=0000001 (1/128)
        min_subnormal=2**-126 * (1 / 128),
    )

    def _get_format_spec(self) -> FormatSpec:
        return self.FORMAT_SPEC

    def _float32_to_bf16_parts(self, f32: float) -> Tuple[int, int, int]:
        """Convert float32 to BF16 parts (sign, exponent, mantissa)"""
        # Get binary representation of float32
        bits = struct.unpack("!I", struct.pack("!f", f32))[0]

        # Extract parts from float32
        sign = (bits >> 31) & 0x1
        exp = (bits >> 23) & 0xFF
        mantissa = (bits >> 16) & 0x7F  # Keep only top 7 bits of mantissa

        return sign, exp, mantissa

    def _bf16_parts_to_float32(self, sign: int, exp: int, mantissa: int) -> float:
        """Convert BF16 parts back to float32"""
        # Construct 32-bit representation
        bits = (sign << 31) | (exp << 23) | (mantissa << 16)
        return struct.unpack("!f", struct.pack("!I", bits))[0]

    def _decimal_to_binary(self, num: float) -> str:
        """Convert decimal number to binary string in BF16 format"""
        if num == 0:
            return "0" * 16

        sign, exp, mantissa = self._float32_to_bf16_parts(num)

        # Handle special cases
        if exp == 0xFF:
            if mantissa == 0:
                # Infinity
                return f"{sign}{'1' * 8}{'0' * 7}"
            else:
                # NaN
                return f"{sign}{'1' * 8}{'1' * 7}"

        # Construct binary string
        return f"{sign:01b}{exp:08b}{mantissa:07b}"

    def _binary_to_decimal(self, binary: str) -> float:
        """Convert binary string in BF16 format to decimal"""
        # Clean binary string
        binary = "".join(c for c in binary if c in "01")

        # Extract components
        sign = int(binary[0], 2)
        exp = int(binary[1:9], 2)
        mantissa = int(binary[9:], 2)

        return self._bf16_parts_to_float32(sign, exp, mantissa)

    @classmethod
    def from_float32(cls, f32: float) -> "BF16":
        """Create BF16 from float32 value"""
        return cls(f32)

    @classmethod
    def inf(cls, sign: bool = False) -> "BF16":
        """Create infinite BF16 value"""
        return cls(binary=f"{'1' if sign else '0'}{'1' * 8}{'0' * 7}")

    @classmethod
    def nan(cls, quiet: bool = True) -> "BF16":
        """
        Create NaN BF16 value
        quiet: If True, creates quiet NaN, else signaling NaN
        """
        if quiet:
            return cls(binary=f"0{'1' * 8}1{'1' * 6}")
        return cls(binary=f"0{'1' * 8}0{'1' * 6}")

    @classmethod
    def max_value(cls) -> "BF16":
        """Create maximum finite value"""
        return cls(binary=f"0{'1' * 7}0{'1' * 7}")

    @classmethod
    def min_normal(cls) -> "BF16":
        """Create minimum normal value"""
        return cls(binary=f"0{'0' * 7}1{'0' * 7}")

    @classmethod
    def min_subnormal(cls) -> "BF16":
        """Create minimum subnormal value"""
        return cls(binary=f"0{'0' * 8}{'0' * 6}1")

    def detailed_breakdown(self) -> dict:
        """
        Provide detailed breakdown of the BF16 number components
        """
        binary = self.binary
        sign_bit = int(binary[0])
        exp_bits = binary[1:9]
        mantissa_bits = binary[9:]

        exp_val = int(exp_bits, 2)
        mantissa_val = int(mantissa_bits, 2)

        return {
            "binary": binary,
            "sign": sign_bit,
            "exponent_bits": exp_bits,
            "exponent_value": (
                exp_val - self._format_spec.bias if exp_val != 0 else "subnormal"
            ),
            "mantissa_bits": mantissa_bits,
            "mantissa_value": mantissa_val,
            "decimal_approx": self.decimal_approx,
            "original_value": self.original_value,
            "is_normal": exp_val != 0 and exp_val != 255,
            "is_subnormal": exp_val == 0 and mantissa_val != 0,
            "is_zero": exp_val == 0 and mantissa_val == 0,
            "is_inf": exp_val == 255 and mantissa_val == 0,
            "is_nan": exp_val == 255 and mantissa_val != 0,
        }
