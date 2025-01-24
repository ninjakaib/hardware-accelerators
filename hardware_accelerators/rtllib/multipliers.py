from typing import Type
import pyrtl
from pyrtl import WireVector

from .utils.common import extract_float_components
from .utils.pipeline import SimplePipeline
from .utils.multiplier_utils import *
from ..dtypes import BaseFloat


def float_multiplier(
    float_a: WireVector,
    float_b: WireVector,
    dtype: Type[BaseFloat],
) -> WireVector:
    e_bits, m_bits = dtype.exponent_bits(), dtype.mantissa_bits()

    sign_out, exp_sum, mant_product = multiplier_stage_2(
        *extract_float_components(float_a, float_b, e_bits, m_bits), m_bits
    )

    leading_zeros, unbiased_exp = multiplier_stage_3(
        exp_sum, mant_product, e_bits, m_bits
    )

    final_exponent, final_mantissa = multiplier_stage_4(
        unbiased_exp, leading_zeros, mant_product, m_bits, e_bits
    )

    result = WireVector(dtype.bitwidth())
    result <<= pyrtl.concat(sign_out, final_exponent, final_mantissa)
    return result


class FloatMultiplierPipelined(SimplePipeline):
    def __init__(
        self,
        float_a: WireVector,
        float_b: WireVector,
        dtype: Type[BaseFloat],
    ):
        self.e_bits = dtype.exponent_bits()
        self.m_bits = dtype.mantissa_bits()
        self._float_a = float_a
        self._float_b = float_b
        self._result = pyrtl.WireVector(dtype.bitwidth())  # , "result")
        super(FloatMultiplierPipelined, self).__init__()

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
        )

    def stage_3(self):
        self.sign_out = self.sign_out
        self.mant_product = self.mant_product
        self.leading_zeros, self.unbiased_exp = multiplier_stage_3(
            self.exp_sum, self.mant_product, self.e_bits, self.m_bits
        )

    def stage_4(self):
        (final_exponent, final_mantissa) = multiplier_stage_4(
            self.unbiased_exp,
            self.leading_zeros,
            self.mant_product,
            self.m_bits,
            self.e_bits,
        )
        self._result <<= pyrtl.concat(self.sign_out, final_exponent, final_mantissa)
