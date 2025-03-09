from typing import Type

import pyrtl
from pyrtl import WireVector, conditional_assignment

from ..dtypes import BaseFloat
from .utils.common import extract_float_components, extract_sign, extract_exponent
from .utils.multiplier_utils import *
from .utils.pipeline import SimplePipeline


def float_multiplier(
    float_a: WireVector,
    float_b: WireVector,
    dtype: Type[BaseFloat],
    fast: bool = False,
) -> WireVector:
    e_bits, m_bits = dtype.exponent_bits(), dtype.mantissa_bits()

    sign_a, sign_b, exp_a, exp_b, mantissa_a, mantissa_b = extract_float_components(
        float_a, float_b, e_bits, m_bits
    )

    # Create a constant zero for exponent comparison.
    zero_exp = pyrtl.Const(0, bitwidth=e_bits)

    # If either exponent is zero, treat the corresponding input as zero.
    is_a_zero = exp_a == zero_exp
    is_b_zero = exp_b == zero_exp
    is_any_zero = is_a_zero | is_b_zero

    # Compute the multiplication normally.
    sign_out, exp_sum, mant_product = multiplier_stage_2(
        sign_a, sign_b, exp_a, exp_b, mantissa_a, mantissa_b, m_bits, fast
    )
    leading_zeros, unbiased_exp = multiplier_stage_3(
        exp_sum, mant_product, e_bits, m_bits, fast
    )
    final_exponent, final_mantissa = multiplier_stage_4(
        unbiased_exp, leading_zeros, mant_product, m_bits, e_bits, fast
    )
    computed_result = pyrtl.concat(sign_out, final_exponent, final_mantissa)

    # If either input is zero, return a zero constant; otherwise return the computed result.
    result = pyrtl.select(
        is_any_zero, pyrtl.Const(0, bitwidth=dtype.bitwidth()), computed_result
    )

    return result


def float_multiplier_fast_unstable(
    float_a: WireVector,
    float_b: WireVector,
    dtype: Type[BaseFloat],
) -> WireVector:
    return float_multiplier(float_a, float_b, dtype, fast=True)


# TODO: add zero detection logic


def float_multiplier_pipelined(
    float_a: WireVector,
    float_b: WireVector,
    dtype: Type[BaseFloat],
) -> WireVector:
    mult = FloatMultiplierPipelined(float_a, float_b, dtype, fast=False)
    return mult._result


def float_multiplier_pipelined_fast_unstable(
    float_a: WireVector,
    float_b: WireVector,
    dtype: Type[BaseFloat],
) -> WireVector:
    mult = FloatMultiplierPipelined(float_a, float_b, dtype, fast=True)
    return mult._result


class FloatMultiplierPipelined(SimplePipeline):
    def __init__(
        self,
        float_a: WireVector,
        float_b: WireVector,
        dtype: Type[BaseFloat],
        fast: bool = False,
    ):
        self._fast = fast
        self.e_bits = dtype.exponent_bits()
        self.m_bits = dtype.mantissa_bits()
        self._float_a = float_a
        self._float_b = float_b
        self._result = pyrtl.WireVector(dtype.bitwidth())  # , "result")
        super().__init__("float_multiplier")

    def stage_1(self):
        (
            self.sign_a,
            self.sign_b,
            self.exp_a,
            self.exp_b,
            self.mantissa_a,
            self.mantissa_b,
        ) = extract_float_components(
            self._float_a, self._float_b, self.e_bits, self.m_bits
        )

    def stage_2(self):
        self.sign_out, self.exp_sum, self.mant_product = multiplier_stage_2(
            self.exp_a,
            self.exp_b,
            self.sign_a,
            self.sign_b,
            self.mantissa_a,
            self.mantissa_b,
            self.m_bits,
            self._fast,
        )

    def stage_3(self):
        self.sign_out = self.sign_out
        self.mant_product = self.mant_product
        self.leading_zeros, self.unbiased_exp = multiplier_stage_3(
            self.exp_sum, self.mant_product, self.e_bits, self.m_bits, self._fast
        )

    def stage_4(self):
        (final_exponent, final_mantissa) = multiplier_stage_4(
            self.unbiased_exp,
            self.leading_zeros,
            self.mant_product,
            self.m_bits,
            self.e_bits,
            self._fast,
        )
        self._result <<= pyrtl.concat(self.sign_out, final_exponent, final_mantissa)
