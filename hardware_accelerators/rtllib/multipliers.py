import pyrtl
from pyrtl import WireVector

from .utils.pipeline import SimplePipeline
from .utils.multiplier_utils import *


def float_multiplier(
    float_a: WireVector,
    float_b: WireVector,
    e_bits: int,
    m_bits: int,
) -> WireVector:

    sign_out, exp_sum, mant_product = stage_2(
        *stage_1(float_a, float_b, e_bits, m_bits)
    )

    leading_zeros, unbiased_exp = stage_3(exp_sum, mant_product)

    final_exponent, final_mantissa = stage_4(unbiased_exp, leading_zeros, mant_product)

    return pyrtl.concat(sign_out, final_exponent, final_mantissa)


class FloatMultiplierPipelined(SimplePipeline):
    def __init__(
        self,
        float_a: WireVector,
        float_b: WireVector,
        e_bits: int,
        m_bits: int,
    ):
        self.e_bits = e_bits
        self.m_bits = m_bits
        self._float_a = float_a
        self._float_b = float_b
        self._result = pyrtl.WireVector(e_bits + m_bits + 1, "result")
        super(FloatMultiplierPipelined, self).__init__()

    def stage_1(self):
        (
            self.sign_a,
            self.sign_b,
            self.exp_a,
            self.exp_b,
            self.mantissa_a,
            self.mantissa_b,
        ) = stage_1(self._float_a, self._float_b, self.e_bits, self.m_bits)

    def stage_2(self):
        (self.sign_out, self.exp_sum, self.mant_product) = stage_2(
            self.exp_a,
            self.exp_b,
            self.sign_a,
            self.sign_b,
            self.mantissa_a,
            self.mantissa_b,
        )

    def stage_3(self):
        self.sign_out = self.sign_out
        self.mant_product = self.mant_product
        (self.leading_zeros, self.unbiased_exp) = stage_3(
            self.exp_sum, self.mant_product
        )

    def stage_4(self):
        (final_exponent, final_mantissa) = stage_4(
            self.unbiased_exp, self.leading_zeros, self.mant_product
        )
        self._result <<= pyrtl.concat(self.sign_out, final_exponent, final_mantissa)
