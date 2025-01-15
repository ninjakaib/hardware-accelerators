from IPython.display import display_svg
from typing import List, Tuple
import random
import torch
import numpy as np
import pandas as pd
from itertools import combinations_with_replacement
from typing import Callable, Any
import pyrtl
from pyrtl.rtllib.libutils import twos_comp_repr, rev_twos_comp_repr
from pyrtl.rtllib import adders
from pyrtl.rtllib import multipliers
from pyrtl import (
    WireVector,
    Const,
    Input,
    Output,
    Register,
    Simulation,
    SimulationTrace,
    reset_working_block,
)


def extract_sign(
    input_a: WireVector, input_b: WireVector, msb: int
) -> Tuple[WireVector, WireVector]:
    sign_a = WireVector(1, name="sign_a")
    sign_b = WireVector(1, name="sign_b")

    sign_a <<= input_a[msb]  # Assuming the MSB is the sign bit
    sign_b <<= input_b[msb]

    return sign_a, sign_b


def extract_exponent(
    input_a: WireVector, input_b: WireVector, e_bits: int
) -> Tuple[WireVector, WireVector]:
    exp_a = WireVector(e_bits, name="exp_a")
    exp_b = WireVector(e_bits, name="exp_b")

    exp_a <<= input_a[-(1 + e_bits) : -1]
    exp_b <<= input_b[-(1 + e_bits) : -1]

    return exp_a, exp_b


def extract_mantissa(
    input_a: WireVector, input_b: WireVector, m_bits: int
) -> Tuple[WireVector, WireVector]:
    mantissa_a = WireVector(m_bits + 1, name="mantissa_a")
    mantissa_b = WireVector(m_bits + 1, name="mantissa_b")

    # Concatenate the implicit leading 1
    mantissa_a <<= pyrtl.concat(Const(1, 1), input_a[:m_bits])
    mantissa_b <<= pyrtl.concat(Const(1, 1), input_b[:m_bits])

    return mantissa_a, mantissa_b


def stage_1(input_a: WireVector, input_b: WireVector, e_bits: int, m_bits: int):
    # Extract components
    sign_a, sign_b = extract_sign(input_a, input_b, e_bits + m_bits)
    exp_a, exp_b = extract_exponent(input_a, input_b, e_bits)
    mantissa_a, mantissa_b = extract_mantissa(input_a, input_b, m_bits)
    return sign_a, sign_b, exp_a, exp_b, mantissa_a, mantissa_b


def stage_2(
    exp_a: WireVector, exp_b: WireVector, sign_a, sign_b, mantissa_a, mantissa_b
):
    # calculate sign using xor
    sign_out = WireVector(1, name="sign_out")
    sign_out <<= sign_a ^ sign_b
    # use adders.cla_adder
    exp_sum = WireVector(len(exp_a) + 1, name="exp_sum")
    exp_sum <<= adders.cla_adder(exp_a, exp_b)
    # csa tree multiplier
    mantissa_product = WireVector(2 * M_BITS + 2, name="mantissa_product")
    mantissa_product <<= multipliers.tree_multiplier(mantissa_a, mantissa_b)
    # return xor for signs and sum of exponents and product of mantissas
    return sign_out, exp_sum, mantissa_product


def enc2(d: WireVector) -> WireVector:
    """
    Encode 2 bits into their leading zero count representation

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
    Merge two encoded vectors in the leading zero count tree

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


def leading_zero_count_16bit(value: WireVector, m_bits: int) -> WireVector:
    """
    Calculate leading zeros for a 8-bit input (for bf16 mantissa addition)

    Args:
        value: 8-bit input WireVector (mantissa with hidden bit and overflow bit)

    Returns:
        4-bit WireVector containing the count of leading zeros (max 8 zeros)
    """
    assert len(value) == m_bits * 2 + 2, "Input must be 8 bits wide"

    # First level: encode pairs of bits (4 pairs total for 8 bits)
    # Results in a list with 4 2-bit WireVectors, indexed from MSB [0] to LSB [-1]
    pairs = pyrtl.chop(value, *[2 for _ in range((2 * m_bits + 2) // 2)])
    encoded_pairs = [enc2(pair) for pair in pairs]

    first_merge = [
        clzi(encoded_pairs[0], encoded_pairs[1], 2),  # First group
        clzi(
            encoded_pairs[2], encoded_pairs[3], 2
        ),  # Last group (handles remaining bits)
        clzi(encoded_pairs[4], encoded_pairs[5], 2),  # First group
        clzi(
            encoded_pairs[6], encoded_pairs[7], 2
        ),  # Last group (handles remaining bits)
    ]

    second_merge = [
        clzi(first_merge[0], first_merge[1], 3),  # First group
        clzi(first_merge[2], first_merge[3], 3),  # Last group (handles remaining bits)
    ]

    final_result = WireVector(5, "lzc_result")
    final_result <<= clzi(second_merge[0], second_merge[1], 4)
    return final_result


def stage_3(exp_sum, mantissa_product):
    # find leading zeros in mantissa product
    leading_zeros = WireVector(E_BITS, name="leading_zeros")
    leading_zeros <<= leading_zero_count_16bit(mantissa_product, M_BITS)

    # adjust exponent
    unbiased_exp = WireVector(E_BITS, name="unbiased_exp")
    unbiased_exp <<= exp_sum - pyrtl.Const(127, E_BITS + 1)

    return leading_zeros, unbiased_exp


def normalize_mantissa_product(mantissa_product, leading_zeros):
    assert mantissa_product.bitwidth == 2 * M_BITS + 2
    assert leading_zeros.bitwidth == 8
    # shift mantissa product to the left by leading zeros
    normalized_mantissa_product = WireVector(
        2 * M_BITS + 2, name="normalized_mantissa_product"
    )
    normalized_mantissa_product <<= pyrtl.shift_left_logical(
        mantissa_product, leading_zeros
    )

    norm_mantissa_msb = WireVector(M_BITS + 1, name="norm_mantissa_msb")
    norm_mantissa_lsb = WireVector(M_BITS + 1, name="norm_mantissa_lsb")

    norm_mantissa_msb <<= normalized_mantissa_product[M_BITS + 1 :]
    norm_mantissa_lsb <<= normalized_mantissa_product[: M_BITS + 1]

    return norm_mantissa_msb, norm_mantissa_lsb


def generate_sgr(
    aligned_mant_lsb: WireVector, m_bits: int
) -> tuple[WireVector, WireVector, WireVector]:
    """
    Generate sticky, guard, and round bits

    Args:
        mant_smaller: original mantissa before shifting (m_bits + 1 wide)
        shift_amount: number of positions shifted right (e_bits wide)
        m_bits: number of mantissa bits (7 for bfloat16)

    Returns:
        sticky_bit, guard_bit, round_bit
    """
    assert len(aligned_mant_lsb) == m_bits + 1

    guard_bit = WireVector(1, name="guard_bit")
    round_bit = WireVector(1, name="round_bit")
    sticky_bit = WireVector(1, name="sticky_bit")

    guard_bit <<= aligned_mant_lsb[m_bits]
    round_bit <<= aligned_mant_lsb[m_bits - 1]
    sticky_bit <<= pyrtl.or_all_bits(aligned_mant_lsb[: m_bits - 1])

    return sticky_bit, guard_bit, round_bit


def rounding_mantissa(norm_mantissa_msb, sticky_bit, guard_bit, round_bit):

    # Round-to-nearest-even logic
    # Round up if:
    # 1. Guard is 1 AND (Round OR Sticky is 1)
    # 2. Guard is 1 AND Round=Sticky=0 AND LSB=1 (tie case, round to even)
    round_up = WireVector(1, "round_up")
    lsb = norm_mantissa_msb[1]  # LSB of normalized mantissa

    round_up <<= (guard_bit & (round_bit | sticky_bit)) | (
        guard_bit & ~round_bit & ~sticky_bit & lsb
    )

    # Add rounding increment
    rounded_mantissa = WireVector(M_BITS + 2, "rounded_mantissa")
    rounded_mantissa <<= norm_mantissa_msb + round_up

    # Check if rounding caused overflow
    extra_increment = WireVector(1, "extra_increment")
    extra_increment <<= rounded_mantissa[M_BITS + 1]  # New carry after rounding

    # Final mantissa (excluding hidden bit)
    final_mantissa = WireVector(M_BITS, "final_mantissa")
    final_mantissa <<= pyrtl.select(
        extra_increment,
        rounded_mantissa[
            1:-1
        ],  # Overflow case: lsb+1 -> msb-1 of rounded mantissa TODO: NEEDS ROUNDING!
        rounded_mantissa[:M_BITS],  # Normal case: take m_bits of rounded mantissa LSBs
    )

    return final_mantissa, extra_increment


def adjust_final_exponent(
    unbiased_exp: WireVector, lzc: WireVector, round_increment: WireVector
) -> WireVector:
    """
    Compute final exponent by subtracting LZC and handling round increment

    Args:
        exp_larger: exponent of larger number (e_bits wide)
        lzc: leading zero count from LZD (4 bits wide)
        round_increment: overflow bit from rounding (1 bit)
        e_bits: number of exponent bits in format

    Returns:
        final_exp: adjusted exponent value (e_bits wide)
    """
    assert len(unbiased_exp) == E_BITS
    assert len(lzc) == 8
    assert len(round_increment) == 1

    # First subtract LZC from larger exponent
    exp_lzc_adjusted = WireVector(E_BITS, "exp_lzc_adjusted")
    exp_lzc_adjusted <<= unbiased_exp - lzc.zero_extended(E_BITS) + 1

    # Then add 1 if rounding caused overflow
    final_exp = WireVector(E_BITS, "final_exp")
    final_exp <<= exp_lzc_adjusted + round_increment

    return final_exp


def stage_4(unbiased_exp, leading_zeros, mantissa_product):
    # normalize mantissa product
    norm_mantissa_msb, norm_mantissa_lsb = normalize_mantissa_product(
        mantissa_product, leading_zeros
    )

    # generate sticky, guard and round bits
    sticky_bit, guard_bit, round_bit = generate_sgr(norm_mantissa_lsb, M_BITS)

    # rounding mantissa
    final_mantissa, extra_increment = rounding_mantissa(
        norm_mantissa_msb, sticky_bit, guard_bit, round_bit
    )

    # adjust final exponent
    final_exponent = adjust_final_exponent(unbiased_exp, leading_zeros, extra_increment)

    return final_exponent, final_mantissa
