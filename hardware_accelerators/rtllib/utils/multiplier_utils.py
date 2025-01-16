from typing import Tuple
import pyrtl
from pyrtl.rtllib import adders
from pyrtl.rtllib import multipliers
from pyrtl import (
    WireVector,
    Const,
)

from .common import leading_zero_counter, generate_sgr

### ===================================================================
### MULTIPLIER STAGE FUNCTIONS
### ===================================================================


def multiplier_stage_2(
    exp_a: WireVector,
    exp_b: WireVector,
    sign_a,
    sign_b,
    mantissa_a,
    mantissa_b,
    m_bits: int,
):
    # calculate sign using xor
    sign_out = WireVector(1, name="sign_out")
    sign_out <<= sign_a ^ sign_b
    # use adders.cla_adder
    exp_sum = WireVector(len(exp_a) + 1, name="exp_sum")
    exp_sum <<= adders.cla_adder(exp_a, exp_b)
    # csa tree multiplier
    mantissa_product = WireVector(2 * m_bits + 2, name="mantissa_product")
    mantissa_product <<= multipliers.tree_multiplier(mantissa_a, mantissa_b)
    # return xor for signs and sum of exponents and product of mantissas
    return sign_out, exp_sum, mantissa_product


def multiplier_stage_3(exp_sum, mantissa_product, e_bits, m_bits: int):
    # find leading zeros in mantissa product
    leading_zeros = WireVector(e_bits, name="leading_zeros")
    leading_zeros <<= leading_zero_counter(mantissa_product, m_bits)

    # adjust exponent
    unbiased_exp = WireVector(e_bits, name="unbiased_exp")
    unbiased_exp <<= exp_sum - pyrtl.Const(2 ** (e_bits - 1) - 1, e_bits + 1)

    return leading_zeros, unbiased_exp


def multiplier_stage_4(unbiased_exp, leading_zeros, mantissa_product, m_bits: int):
    # normalize mantissa product
    norm_mantissa_msb, norm_mantissa_lsb = normalize_mantissa_product(
        mantissa_product, leading_zeros
    )

    # generate sticky, guard and round bits
    sticky_bit, guard_bit, round_bit = generate_sgr(norm_mantissa_lsb, m_bits)

    # rounding mantissa
    final_mantissa, extra_increment = rounding_mantissa(
        norm_mantissa_msb, sticky_bit, guard_bit, round_bit
    )

    # adjust final exponent
    final_exponent = adjust_final_exponent(unbiased_exp, leading_zeros, extra_increment)

    return final_exponent, final_mantissa


### ===================================================================
### MULTIPLIER HELPER FUNCTIONS
### ===================================================================


def normalize_mantissa_product(mantissa_product, leading_zeros, m_bits: int):
    assert mantissa_product.bitwidth == 2 * m_bits + 2
    assert leading_zeros.bitwidth == 8
    # shift mantissa product to the left by leading zeros
    normalized_mantissa_product = WireVector(
        2 * m_bits + 2, name="normalized_mantissa_product"
    )
    normalized_mantissa_product <<= pyrtl.shift_left_logical(
        mantissa_product, leading_zeros
    )

    norm_mantissa_msb = WireVector(m_bits + 1, name="norm_mantissa_msb")
    norm_mantissa_lsb = WireVector(m_bits + 1, name="norm_mantissa_lsb")

    norm_mantissa_msb <<= normalized_mantissa_product[m_bits + 1 :]
    norm_mantissa_lsb <<= normalized_mantissa_product[: m_bits + 1]

    return norm_mantissa_msb, norm_mantissa_lsb


def rounding_mantissa(norm_mantissa_msb, sticky_bit, guard_bit, round_bit, m_bits: int):

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
    rounded_mantissa = WireVector(m_bits + 2, "rounded_mantissa")
    rounded_mantissa <<= norm_mantissa_msb + round_up

    # Check if rounding caused overflow
    extra_increment = WireVector(1, "extra_increment")
    extra_increment <<= rounded_mantissa[m_bits + 1]  # New carry after rounding

    # Final mantissa (excluding hidden bit)
    final_mantissa = WireVector(m_bits, "final_mantissa")
    final_mantissa <<= pyrtl.select(
        extra_increment,
        rounded_mantissa[
            1:-1
        ],  # Overflow case: lsb+1 -> msb-1 of rounded mantissa TODO: NEEDS ROUNDING!
        rounded_mantissa[:m_bits],  # Normal case: take m_bits of rounded mantissa LSBs
    )

    return final_mantissa, extra_increment


def adjust_final_exponent(
    unbiased_exp: WireVector, lzc: WireVector, round_increment: WireVector, e_bits: int
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
    assert len(unbiased_exp) == e_bits
    assert len(lzc) == 8
    assert len(round_increment) == 1

    # First subtract LZC from larger exponent
    exp_lzc_adjusted = WireVector(e_bits, "exp_lzc_adjusted")
    exp_lzc_adjusted <<= unbiased_exp - lzc.zero_extended(e_bits) + 1

    # Then add 1 if rounding caused overflow
    final_exp = WireVector(e_bits, "final_exp")
    final_exp <<= exp_lzc_adjusted + round_increment

    return final_exp
