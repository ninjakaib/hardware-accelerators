from typing import Tuple
import pyrtl
from pyrtl import WireVector, Const


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


def leading_zero_counter(value: WireVector, m_bits: int) -> WireVector:
    """
    Calculate leading zeros for a 8-bit input (for bf16 mantissa addition)

    Args:
        value: 8-bit input WireVector (mantissa with hidden bit and overflow bit)

    Returns:
        4-bit WireVector containing the count of leading zeros (max 8 zeros)
    """
    assert len(value) == m_bits + 1, f"Input must be 8 bits wide, {len(value)=}"

    # First level: encode pairs of bits (4 pairs total for 8 bits)
    # Results in a list with 4 2-bit WireVectors, indexed from MSB [0] to LSB [-1]
    pairs = pyrtl.chop(value, *[2 for _ in range((m_bits + 1) // 2)])
    encoded_pairs = [enc2(pair) for pair in pairs]

    if m_bits == 7:  # bfloat16
        first_merge = [
            clzi(encoded_pairs[0], encoded_pairs[1], 2),  # First group
            clzi(
                encoded_pairs[2], encoded_pairs[3], 2
            ),  # Last group (handles remaining bits)
        ]
        final_result = WireVector(4)  # , "lzc_result")
        final_result <<= clzi(first_merge[0], first_merge[1], 3)
        return final_result

    elif m_bits == 3:  # float8
        final_result = WireVector(4)  # , "lzc_result")
        final_result <<= clzi(encoded_pairs[0], encoded_pairs[1], 2)
        return final_result

    else:
        raise Warning(
            f"Leading zero counter not implemented for float type with {m_bits} mantissa bits"
        )


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
