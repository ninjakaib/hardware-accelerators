from typing import Tuple

import pyrtl
from pyrtl import Const, WireVector

from .common import generate_sgr, clzi, enc2

### ===================================================================
### ADDER STAGE FUNCTIONS
### ===================================================================


def adder_stage_2(
    sign_a: WireVector,
    sign_b: WireVector,
    exp_a: WireVector,
    exp_b: WireVector,
    mant_a: WireVector,
    mant_b: WireVector,
    e_bits: int,
    m_bits: int,
):
    assert len(sign_a) == len(sign_b) == 1
    assert len(exp_a) == len(exp_b) == e_bits
    assert len(mant_a) == len(mant_b) == m_bits + 1

    sign_xor = WireVector(1)  # , name="sign_xor")
    exp_diff = WireVector(e_bits + 1)  # , name="exp_diff")
    signed_shift = WireVector(e_bits + 1)  # , name="signed_shift")
    exp_larger = WireVector(e_bits)  # , name="exp_larger")
    mant_smaller = WireVector(m_bits + 1)  # , name="mant_smaller")
    mant_larger = WireVector(m_bits + 1)  # , name="mant_larger")

    sign_xor <<= sign_a ^ sign_b

    exp_diff <<= exp_a - exp_b  # This can be negative, indicating which is larger
    exp_larger <<= pyrtl.mux(exp_diff[e_bits], exp_a, exp_b)
    signed_shift <<= pyrtl.mux(
        exp_diff[e_bits],
        exp_diff[:e_bits],
        pyrtl.concat(exp_diff[e_bits], (~exp_diff[:e_bits] + 1)[:e_bits]),
    )

    # Select the smaller mantissa to align and the larger to keep unchanged
    mant_smaller <<= pyrtl.mux(exp_diff[e_bits], mant_b, mant_a)
    mant_larger <<= pyrtl.mux(exp_diff[e_bits], mant_a, mant_b)

    return sign_xor, exp_larger, signed_shift, mant_smaller, mant_larger


def adder_stage_3(
    mant_smaller: WireVector, shift_amount: WireVector, m_bits: int, e_bits: int
):
    """
    Combine alignment and SGR generation
    """
    aligned_mant_msb, aligned_mant_lsb = align_mantissa(
        mant_smaller, shift_amount, m_bits, e_bits
    )
    sticky_bit, guard_bit, round_bit = generate_sgr(aligned_mant_lsb, m_bits)

    return aligned_mant_msb, sticky_bit, guard_bit, round_bit


def adder_stage_4(
    mant_aligned: WireVector,
    mant_unchanged: WireVector,
    sign_xor: WireVector,
    m_bits: int,
):
    """
    Add/sub the mantissas based on input signs and count the leading zeros in the result

    Args:
        mant_aligned: aligned mantissa (m_bits + 1 wide)
        mant_unchanged: unchanged mantissa (m_bits + 1 wide)
        sign_xor: XOR of input signs
        m_bits: number of mantissa bits

    Returns:
        result_mantissa: result of add/sub (m_bits + 2 wide to handle overflow)
        is_negative: whether the result is negative
        lzc: leading zero count (4 bits wide to count up to 9 zeros)
    """

    mantissa_sum, is_neg = add_sub_mantissas(
        mant_aligned, mant_unchanged, sign_xor, m_bits
    )
    lzc = leading_zero_detector_module(mantissa_sum, m_bits)
    return mantissa_sum, is_neg, lzc


def adder_stage_5(
    abs_mantissa: WireVector,
    sticky_bit: WireVector,
    guard_bit: WireVector,
    round_bit: WireVector,
    lzc: WireVector,
    exp_larger: WireVector,
    sign_a: WireVector,
    sign_b: WireVector,
    exp_diff: WireVector,
    is_neg: WireVector,
    e_bits: int,
    m_bits: int,
) -> tuple[WireVector, WireVector, WireVector]:
    """
    Stage 5: Normalization, Rounding, and Final Adjustments

    Args:
        abs_mantissa: absolute value of mantissa sum (m_bits + 2 wide)
        sticky_bit: sticky bit from alignment
        guard_bit: guard bit from alignment
        round_bit: round bit from alignment
        lzc: leading zero count from LZD (4 bits wide)
        exp_larger: larger input exponent (e_bits wide)
        sign_a: sign of first input
        sign_b: sign of second input
        exp_diff: exponent difference (e_bits + 1 wide)
        is_neg: sign from mantissa operation
        e_bits: number of exponent bits in format
        m_bits: number of mantissa bits in format

    Returns:
        final_sign: computed sign for result
        final_exp: adjusted exponent value
        final_mantissa: normalized and rounded mantissa
    """
    assert len(abs_mantissa) == m_bits + 2
    assert len(sticky_bit) == len(guard_bit) == len(round_bit) == 1
    assert len(lzc) == 4
    assert len(exp_larger) == e_bits, f"{len(exp_larger)=}, must be {e_bits}"
    assert len(sign_a) == len(sign_b) == len(is_neg) == 1
    assert len(exp_diff) == e_bits + 1

    # 1. Normalize and round mantissa
    norm_mantissa, round_inc = normalize_and_round(
        abs_mantissa, sticky_bit, guard_bit, round_bit, lzc, m_bits
    )

    # 2. Adjust exponent
    final_exp = adjust_final_exponent(exp_larger, lzc, round_inc, e_bits)

    # 3. Determine final sign
    final_sign = detect_final_sign(sign_a, sign_b, exp_diff, is_neg, e_bits)

    return final_sign, final_exp, norm_mantissa


### ===================================================================
### HELPER FUNCTIONS
### ===================================================================


def calculate_exponent_difference(exp_a: WireVector, exp_b: WireVector):
    assert len(exp_a) == len(exp_b)
    exp_diff = WireVector(len(exp_a) + 1)  # , name="exp_diff")
    exp_greater = WireVector(len(exp_a))  # , name="exp_greater")

    exp_diff <<= pyrtl.signed_sub(
        exp_a, exp_b
    )  # This can be negative, indicating which is larger
    exp_greater <<= pyrtl.mux(exp_diff[8], exp_a, exp_b)
    return exp_diff, exp_greater


def align_mantissa(
    mant_smaller: WireVector, shift_amount: WireVector, m_bits: int, e_bits: int
) -> WireVector:
    """
    Align the smaller mantissa by shifting right

    Args:
        mant_smaller: mantissa to be shifted (m_bits + 1 wide)
        shift_amount: number of positions to shift right (e_bits wide)
        m_bits: number of mantissa bits (7 for bfloat16)

    Returns:
        aligned_mantissa_msb, aligned_mantissa_lsb: shifted mantissa
    """
    assert len(mant_smaller) == m_bits + 1
    assert len(shift_amount) == e_bits

    # Detect if shift amount exceeds mantissa width
    max_useful_shift = 2 * (m_bits + 1)

    # Clamp shift amount to maximum useful value
    clamped_shift = WireVector(e_bits)  # , "clamped_shift")
    clamped_shift <<= pyrtl.select(
        shift_amount > max_useful_shift,
        max_useful_shift,
        shift_amount,
    )

    # Perform the right shift
    extended_mantissa = pyrtl.concat(mant_smaller, pyrtl.Const(0, m_bits + 1))

    aligned_mantissa = WireVector(max_useful_shift)  # , name="aligned_mantissa")
    assert len(extended_mantissa) == len(aligned_mantissa)

    aligned_mantissa <<= pyrtl.shift_right_logical(extended_mantissa, clamped_shift)
    return pyrtl.chop(aligned_mantissa, *[m_bits + 1] * 2)  # type: ignore


def add_sub_mantissas(
    mant_aligned: WireVector,
    mant_unchanged: WireVector,
    sign_xor: WireVector,
    m_bits: int,
) -> tuple[WireVector, WireVector]:
    """
    Perform addition or subtraction on mantissas based on signs

    Args:
        mant_aligned: aligned mantissa (m_bits + 1 wide)
        mant_unchanged: unchanged mantissa (m_bits + 1 wide)
        sign_xor: result of XOR on the input signs
        m_bits: number of mantissa bits

    Returns:
        result_mantissa: result of add/sub (m_bits + 2 wide to handle overflow)
        is_negative: whether the result is negative
    """
    assert len(mant_aligned) == len(mant_unchanged) == m_bits + 1
    assert len(sign_xor) == 1

    raw_result = WireVector(m_bits + 3)  # , "raw_result")
    with pyrtl.conditional_assignment:
        with sign_xor:
            raw_result |= mant_unchanged.zero_extended(
                m_bits + 2
            ) - mant_aligned.zero_extended(m_bits + 2)
        with pyrtl.otherwise:
            raw_result |= mant_unchanged.zero_extended(
                m_bits + 2
            ) + mant_aligned.zero_extended(m_bits + 2)

    # Detect if result is negative
    is_negative = WireVector(1)  # , "is_negative")
    is_negative <<= raw_result[m_bits + 2]  # MSB indicates sign

    # If result is negative, convert back to positive
    abs_result = WireVector(m_bits + 2)  # , "abs_result")
    abs_result <<= pyrtl.select(is_negative, ~raw_result + 1, raw_result)

    return abs_result, is_negative


def adder_leading_zero_counter(value: WireVector, m_bits: int) -> WireVector:
    """
    Calculate leading zeros for a 8-bit input (for bf16 mantissa addition)

    Args:
        value: 8-bit input WireVector (mantissa with hidden bit and overflow bit)

    Returns:
        4-bit WireVector containing the count of leading zeros (max 8 zeros)
    """
    assert (
        len(value) == m_bits + 1
    ), f"Input must be {m_bits + 1} bits wide, got {len(value)}"
    print(f"adder_leading_zero_counter() called from adder_utils.py on line 261")

    # First level: encode pairs of bits (4 pairs total for 8 bits)
    # Results in a list with 4 2-bit WireVectors, indexed from MSB [0] to LSB [-1]
    pairs = pyrtl.chop(value, *[2 for _ in range((m_bits + 1) // 2)])
    encoded_pairs = [enc2(pair) for pair in pairs]
    print(f"{len(encoded_pairs)=}")

    if m_bits == 7:  # bfloat16
        print(f"{m_bits=}")
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


def leading_zero_detector_module(mantissa_sum: WireVector, m_bits: int) -> WireVector:
    """
    Calculate leading zeros in mantissa sum, accounting for overflow bit

    Args:
        mantissa_sum: sum of mantissas (m_bits + 2 wide to handle overflow)
        m_bits: number of mantissa bits in original format

    Returns:
        lzc: leading zero count (4 bits wide to count up to 9 zeros)

    The module checks the overflow (carry) bit to determine if normalization is needed:
    - If carry_bit is 1: no leading zeros (return 0)
    - If carry_bit is 0: count leading zeros in remaining bits and add 1
    """
    assert len(mantissa_sum) == m_bits + 2, f"Input must be {m_bits+2} bits wide"

    # Use the carry bit from the sum result to determine if we use the LZC or not
    carry_bit = mantissa_sum[-1]

    lzc = WireVector(4)  # , "leading_zero_count")
    lzc <<= pyrtl.select(
        carry_bit, 0, adder_leading_zero_counter(mantissa_sum[:-1], m_bits) + 1
    )
    return lzc


def detect_final_sign(
    sign_a: WireVector,
    sign_b: WireVector,
    exp_diff: WireVector,
    is_neg: WireVector,
    e_bits: int,
) -> WireVector:
    """
    Determine the final sign of the floating point addition result

    Args:
        sign_a: sign bit of first number
        sign_b: sign bit of second number
        exp_diff: difference between exponents (exp_a - exp_b)
        is_neg: sign from mantissa addition/subtraction
        e_bits: number of exponent bits

    Returns:
        final_sign: computed sign bit for the result
    """
    assert len(sign_a) == len(sign_b) == len(is_neg) == 1
    assert len(exp_diff) == e_bits + 1  # Signed difference

    final_sign = WireVector(1)  # , "final_sign")

    # Check if signs are the same
    same_signs = ~(sign_a ^ sign_b)

    # For same signs case, use either input sign (they're the same)
    # For different signs case, use magnitude comparison
    with pyrtl.conditional_assignment:
        with same_signs:
            final_sign |= sign_a
        with pyrtl.otherwise:
            # If exponents are equal (exp_diff == 0), use mantissa comparison
            # Otherwise, use sign of number with larger exponent
            with exp_diff[: e_bits - 1] == 0:
                final_sign |= is_neg
            with exp_diff[e_bits]:  # exp_diff is negative, meaning exp_b > exp_a
                final_sign |= sign_b
            with pyrtl.otherwise:  # exp_diff is positive, meaning exp_a > exp_b
                final_sign |= sign_a

    return final_sign


def normalize_and_round(
    abs_mantissa: WireVector,
    sticky_bit: WireVector,
    guard_bit: WireVector,
    round_bit: WireVector,
    lzc: WireVector,
    m_bits: int,
) -> tuple[WireVector, WireVector]:
    """
    Normalize and round the mantissa result

    Args:
        abs_mantissa: absolute value of mantissa sum (m_bits + 2 wide)
        sticky_bit: sticky bit from alignment
        guard_bit: guard bit from alignment
        round_bit: round bit from alignment
        lzc: leading zero count from LZD (4 bits wide)
        m_bits: number of mantissa bits in format

    Returns:
        norm_mantissa: normalized and rounded mantissa (m_bits wide)
        extra_increment: whether rounding caused an overflow requiring exponent adjustment
    """
    assert len(abs_mantissa) == m_bits + 2
    assert len(sticky_bit) == len(guard_bit) == len(round_bit) == 1
    assert len(lzc) == 4

    # Normalize by shifting left according to LZD
    norm_shift = WireVector(m_bits + 2)  # , "norm_shift")
    norm_shift <<= pyrtl.shift_left_logical(abs_mantissa, lzc.zero_extended(m_bits + 2))

    # Round-to-nearest-even logic
    # Round up if:
    # 1. Guard is 1 AND (Round OR Sticky is 1)
    # 2. Guard is 1 AND Round=Sticky=0 AND LSB=1 (tie case, round to even)
    round_up = WireVector(1)  # , "round_up")
    lsb = norm_shift[1]  # LSB of normalized mantissa

    round_up <<= (guard_bit & (round_bit | sticky_bit)) | (
        guard_bit & ~round_bit & ~sticky_bit & lsb
    )

    # Add rounding increment
    rounded_mantissa = WireVector(m_bits + 2)  # , "rounded_mantissa")
    rounded_mantissa <<= norm_shift + round_up

    # Check if rounding caused overflow
    extra_increment = WireVector(1)  # , "extra_increment")
    extra_increment <<= rounded_mantissa[m_bits + 1]  # New carry after rounding

    # Final mantissa (excluding hidden bit)
    final_mantissa = WireVector(m_bits)  # , "final_mantissa")
    final_mantissa <<= pyrtl.select(
        extra_increment,
        rounded_mantissa[
            1:-1
        ],  # Overflow case: lsb+1 -> msb-1 of rounded mantissa TODO: NEEDS ROUNDING!
        rounded_mantissa[:m_bits],  # Normal case: take m_bits of rounded mantissa LSBs
    )

    return final_mantissa, extra_increment


def adjust_final_exponent(
    exp_larger: WireVector, lzc: WireVector, round_increment: WireVector, e_bits: int
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
    assert len(exp_larger) == e_bits
    assert len(lzc) == 4
    assert len(round_increment) == 1

    # First subtract LZC from larger exponent
    lzc_adjusted = WireVector(e_bits)  # , "lzc_adjusted")
    lzc_adjusted <<= exp_larger - lzc.zero_extended(e_bits)

    # Then add 1 if rounding caused overflow
    final_exp = WireVector(e_bits)  # , "final_exp")
    final_exp <<= lzc_adjusted + round_increment

    return final_exp
