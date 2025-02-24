from typing import Tuple
from ...dtypes import BaseFloat, Float8, Float16, Float32
import pyrtl
from pyrtl import Const, WireVector


def convert_float_format(
    input_wire: pyrtl.WireVector,
    input_dtype: type[BaseFloat],
    output_dtype: type[BaseFloat],
) -> pyrtl.WireVector:
    """
    Convert a WireVector representing a floating point number from one format to another.
    Only allows upcasting (converting from smaller to larger formats).

    Args:
        input_wire: WireVector containing the input floating point number
        input_dtype: Input floating point format class (subclass of BaseFloat)
        output_dtype: Output floating point format class (subclass of BaseFloat)

    Returns:
        WireVector containing the converted floating point number

    Raises:
        ValueError: If attempting to downcast or if formats are incompatible
    """
    # Validate input and output types
    if not issubclass(input_dtype, BaseFloat) or not issubclass(
        output_dtype, BaseFloat
    ):
        raise ValueError("Input and output types must be BaseFloat subclasses")

    # Validate wire width matches input format
    if len(input_wire) != input_dtype.bitwidth():
        raise ValueError(
            f"Input wire width {len(input_wire)} does not match format width {input_dtype.bitwidth()}"
        )

    # # Check for valid upcasting
    # if output_dtype.bitwidth() < input_dtype.bitwidth():
    #     raise ValueError("Cannot downcast to smaller format")

    # If same format, return input wire directly
    if input_dtype == output_dtype:
        return input_wire

    # Extract components from input using the input format's specs
    sign = input_wire[input_dtype.bitwidth() - 1]
    exp = input_wire[input_dtype.mantissa_bits() : input_dtype.bitwidth() - 1]
    mantissa = input_wire[: input_dtype.mantissa_bits()]

    # Calculate the bias difference between formats
    bias_diff = output_dtype.bias() - input_dtype.bias()

    result = pyrtl.WireVector(output_dtype.bitwidth())

    # Create new exponent with adjusted bias
    new_exp = pyrtl.WireVector(output_dtype.exponent_bits())

    # Create new mantissa with padding
    new_mantissa = pyrtl.WireVector(output_dtype.mantissa_bits())
    # Pad with zeros on the right

    # Handle bias adjustment for normal numbers
    if input_dtype == Float8:
        # Special handling for Float8
        is_nan = pyrtl.and_all_bits(input_wire[:7])

        with pyrtl.conditional_assignment:
            # If input is zero (all exp bits are 0)
            with exp == 0:
                new_exp |= 0
                new_mantissa |= 0
            # If input is nan (all bits are 1)
            with is_nan:
                new_exp |= 2 ** output_dtype.exponent_bits() - 1
                new_mantissa |= 2 ** output_dtype.mantissa_bits() - 1
            # Normal numbers - adjust bias
            with pyrtl.otherwise:
                new_exp |= exp + bias_diff
                new_mantissa |= pyrtl.concat(
                    mantissa,
                    pyrtl.Const(
                        0, output_dtype.mantissa_bits() - input_dtype.mantissa_bits()
                    ),
                )

    elif input_dtype == Float16:
        is_nan = pyrtl.and_all_bits(input_wire[:15])

        with pyrtl.conditional_assignment:
            # If input is zero (all exp bits are 0)
            with exp == 0:
                new_exp |= 0
                new_mantissa |= 0
            # If input is nan (all bits are 1)
            with is_nan:
                new_exp |= 2 ** output_dtype.exponent_bits() - 1
                new_mantissa |= 2 ** output_dtype.mantissa_bits() - 1
            # Normal numbers - adjust bias
            with pyrtl.otherwise:
                new_exp |= exp + bias_diff
                truncate_bits = (
                    input_dtype.mantissa_bits() - output_dtype.mantissa_bits()
                )
                truncated_mantissa = mantissa[
                    truncate_bits:
                ]  # Slice [3:10] for conversion to 7-bit mantissa
                new_mantissa |= truncated_mantissa

    elif input_dtype == Float32:
        is_nan = pyrtl.and_all_bits(input_wire[:31])

        with pyrtl.conditional_assignment:
            # If input is zero (all exp bits are 0)
            with exp == 0:
                new_exp |= 0
                new_mantissa |= 0
            # If input is nan (all bits are 1)
            with is_nan:
                new_exp |= 2 ** output_dtype.exponent_bits() - 1
                new_mantissa |= 2 ** output_dtype.mantissa_bits() - 1
            # Normal numbers - adjust bias
            with pyrtl.otherwise:
                new_exp |= exp + bias_diff
                truncate_bits = (
                    input_dtype.mantissa_bits() - output_dtype.mantissa_bits()
                )
                truncated_mantissa = mantissa[
                    truncate_bits:
                ]  # Slice [3:10] for conversion to 7-bit mantissa
                new_mantissa |= truncated_mantissa

    else:
        with pyrtl.conditional_assignment:
            # If input is zero (all exp bits are 0)
            with exp == 0:
                new_exp |= 0
            # If input is inf/nan (all exp bits are 1)
            with exp == (2 ** input_dtype.exponent_bits() - 1):
                new_exp |= 2 ** output_dtype.exponent_bits() - 1
            # Normal numbers - adjust bias
            with pyrtl.otherwise:
                new_exp |= exp + bias_diff

    # Combine components into final result
    result <<= pyrtl.concat(sign, new_exp, new_mantissa)

    return result


def extract_float_components(
    input_a: WireVector, input_b: WireVector, e_bits: int, m_bits: int
):
    # Extract components
    sign_a, sign_b = extract_sign(input_a, input_b, e_bits + m_bits)
    exp_a, exp_b = extract_exponent(input_a, input_b, e_bits)
    mantissa_a, mantissa_b = extract_mantissa(input_a, input_b, m_bits)
    return sign_a, sign_b, exp_a, exp_b, mantissa_a, mantissa_b


def extract_sign(
    input_a: WireVector, input_b: WireVector, msb: int
) -> Tuple[WireVector, WireVector]:
    """
    Extract sign bits from two input vectors.

    Args:
        input_a: First input vector
        input_b: Second input vector
        msb: Most significant bit position (total width - 1)

    Returns:
        Tuple of sign bits for both inputs
    """
    sign_a = WireVector(1)  # , name="sign_a")
    sign_b = WireVector(1)  # , name="sign_b")

    sign_a <<= input_a[msb]  # Assuming the MSB is the sign bit
    sign_b <<= input_b[msb]

    return sign_a, sign_b


def extract_exponent(
    input_a: WireVector, input_b: WireVector, e_bits: int
) -> Tuple[WireVector, WireVector]:
    """
    Extract exponent bits from two input vectors.

    Args:
        input_a: First input vector
        input_b: Second input vector
        e_bits: Number of exponent bits

    Returns:
        Tuple of exponent bits for both inputs
    """
    exp_a = WireVector(e_bits)  # , name="exp_a")
    exp_b = WireVector(e_bits)  # , name="exp_b")

    exp_a <<= input_a[-(1 + e_bits) : -1]
    exp_b <<= input_b[-(1 + e_bits) : -1]

    return exp_a, exp_b


def extract_mantissa(
    input_a: WireVector, input_b: WireVector, m_bits: int
) -> Tuple[WireVector, WireVector]:
    """
    Extract mantissa bits from two input vectors and add implicit leading 1.

    Args:
        input_a: First input vector
        input_b: Second input vector
        m_bits: Number of mantissa bits

    Returns:
        Tuple of mantissa bits (with implicit 1) for both inputs
    """
    mantissa_a = WireVector(m_bits + 1)  # , name="mantissa_a")
    mantissa_b = WireVector(m_bits + 1)  # , name="mantissa_b")

    # Concatenate the implicit leading 1
    mantissa_a <<= pyrtl.concat(Const(1, 1), input_a[:m_bits])
    mantissa_b <<= pyrtl.concat(Const(1, 1), input_b[:m_bits])

    return mantissa_a, mantissa_b


def enc2(d: WireVector) -> WireVector:
    """
    Encode 2 bits into their leading zero count representation.

    Args:
        d: 2-bit input WireVector

    Returns:
        2-bit WireVector encoding the leading zeros:
        00 -> 10 (2 leading zeros)
        01 -> 01 (1 leading zero)
        10 -> 00 (0 leading zeros)
        11 -> 00 (0 leading zeros)
    """
    result = WireVector(2)
    with pyrtl.conditional_assignment:
        with d == 0b00:
            result |= 0b10
        with d == 0b01:
            result |= 0b01
        with pyrtl.otherwise:
            result |= 0b00
    return result


def clzi(left: WireVector, right: WireVector, n: int) -> WireVector:
    """
    Merge two encoded vectors in the leading zero count tree.

    Args:
        left: Left section to process
        right: Right section to process
        n: Size of each pair

    Returns:
        Merged and encoded output vector
    """
    assert len(left) == len(right) == n
    result = WireVector(n + 1)

    left_msb, right_msb = left[-1], right[-1]

    with pyrtl.conditional_assignment:
        with left_msb & right_msb:  # both sides start with 1
            result |= Const(1 << n, bitwidth=n + 1)  # n leading zeros
        with left_msb:  # left starts with 1
            result |= pyrtl.concat(
                Const(0b01, bitwidth=2), right[: n - 1]  # 01  # right[:MSB-1]
            )
        with pyrtl.otherwise:  # left starts with 0
            result |= pyrtl.concat(
                Const(0, bitwidth=1), left  # 0  # Rest from left section
            )

    return result


def generate_sgr(
    aligned_mant_lsb: WireVector, m_bits: int
) -> tuple[WireVector, WireVector, WireVector]:
    """
    Generate sticky, guard, and round bits.

    Args:
        aligned_mant_lsb: Lower bits of aligned mantissa (m_bits + 1 wide)
        m_bits: Number of mantissa bits

    Returns:
        Tuple of (sticky_bit, guard_bit, round_bit)
    """
    assert len(aligned_mant_lsb) == m_bits + 1

    guard_bit = WireVector(1)  # , name="guard_bit")
    round_bit = WireVector(1)  # , name="round_bit")
    sticky_bit = WireVector(1)  # , name="sticky_bit")

    guard_bit <<= aligned_mant_lsb[m_bits]
    round_bit <<= aligned_mant_lsb[m_bits - 1]
    sticky_bit <<= pyrtl.or_all_bits(aligned_mant_lsb[: m_bits - 1])

    return sticky_bit, guard_bit, round_bit
