from typing import Type

import pyrtl
from pyrtl import WireVector, conditional_assignment
from pyrtl.rtllib.adders import carrysave_adder, kogge_stone, fast_group_adder

from ..dtypes import BaseFloat, Float8
from .utils.lmul_utils import get_combined_offset, lmul_offset_rtl


def lmul(float_a: WireVector, float_b: WireVector, dtype: Type[BaseFloat], fast=False):
    e_bits, m_bits = dtype.exponent_bits(), dtype.mantissa_bits()
    em_bits = e_bits + m_bits
    sign_a = float_a[em_bits]
    sign_b = float_b[em_bits]
    exp_a = float_a[m_bits:-1]
    exp_b = float_b[m_bits:-1]
    exp_mantissa_a = float_a[:em_bits]
    exp_mantissa_b = float_b[:em_bits]

    zero_or_subnormal = WireVector(1)
    final_sum = WireVector(em_bits + 2)
    carry_msb = WireVector(2)
    fp_out = WireVector(dtype.bitwidth())

    OFFSET_MINUS_BIAS = lmul_offset_rtl(dtype)
    MAX_VALUE = pyrtl.Const(dtype.binary_max(), bitwidth=em_bits)

    if fast:
        final_sum <<= carrysave_adder(
            exp_mantissa_a, exp_mantissa_b, OFFSET_MINUS_BIAS, final_adder=kogge_stone
        )
    else:
        final_sum <<= exp_mantissa_a + exp_mantissa_b + OFFSET_MINUS_BIAS

    carry_msb <<= final_sum[em_bits:]
    zero_or_subnormal <<= ~pyrtl.or_all_bits(exp_a) | ~pyrtl.or_all_bits(exp_b)

    with conditional_assignment:
        with zero_or_subnormal:
            fp_out |= 0
        with carry_msb == 0:
            fp_out |= 0
        with carry_msb == 1:
            fp_out |= pyrtl.concat(sign_a ^ sign_b, final_sum[:em_bits])
        with pyrtl.otherwise:
            fp_out |= pyrtl.concat(sign_a ^ sign_b, MAX_VALUE)

    return fp_out


def lmul_simple(float_a: WireVector, float_b: WireVector, dtype: Type[BaseFloat]):
    return lmul(float_a, float_b, dtype, fast=False)


def lmul_fast(float_a: WireVector, float_b: WireVector, dtype: Type[BaseFloat]):
    return lmul(float_a, float_b, dtype, fast=True)


def lmul_pipelined(
    float_a: WireVector,
    float_b: WireVector,
    dtype: Type[BaseFloat],
) -> WireVector:
    mult = LmulPipelined(float_a, float_b, dtype)
    return mult.output_reg


def lmul_pipelined_fast(
    float_a: WireVector,
    float_b: WireVector,
    dtype: Type[BaseFloat],
) -> WireVector:
    mult = LmulPipelined(float_a, float_b, dtype, fast=True)
    return mult.output_reg


class LmulPipelined:
    def __init__(
        self,
        float_a: WireVector,
        float_b: WireVector,
        dtype: Type[BaseFloat],
        fast: bool = False,
    ):
        self.e_bits = dtype.exponent_bits()
        self.m_bits = dtype.mantissa_bits()
        self.em_bits = dtype.bitwidth() - 1
        self._fast = fast

        # Inputs and Outputs
        assert (
            len(float_a) == len(float_b) == self.em_bits + 1
        ), "Input bitwidths must match e_bits + m_bits + 1"
        self.fp_a = float_a
        self.fp_b = float_b

        # Constants
        self.OFFSET_MINUS_BIAS = pyrtl.Const(
            get_combined_offset(self.e_bits, self.m_bits, twos_comp=True),
            bitwidth=self.em_bits,
        )

        self.MAX_VALUE = pyrtl.Const(2**self.em_bits - 1, bitwidth=self.em_bits)

        if dtype == Float8:
            self.MAX_VALUE = pyrtl.Const(0x7F, 7)

        # Pipeline Registers
        # Stage 0 -> 1
        self.reg_fp_a = pyrtl.Register(self.em_bits + 1)  # , "reg_fp_a")
        self.reg_fp_b = pyrtl.Register(self.em_bits + 1)  # , "reg_fp_b")

        # Stage 1 -> 2
        self.reg_sign = pyrtl.Register(1)  # , "reg_sign")
        self.reg_final_sum = pyrtl.Register(self.em_bits + 2)  # , "reg_final_sum")

        # Stage 2 -> output
        self.output_reg = pyrtl.Register(self.em_bits + 1)  # , "reg_output")

        # Build pipeline
        self._build_pipeline()

    def stage0_input(self):
        """Input registration stage"""
        self.reg_fp_a.next <<= self.fp_a
        self.reg_fp_b.next <<= self.fp_b

    def stage1_split_and_add(self):
        """Split inputs and perform additions"""
        # Split registered inputs
        sign_a = self.reg_fp_a[self.em_bits]
        sign_b = self.reg_fp_b[self.em_bits]
        exp_mantissa_a = self.reg_fp_a[0 : self.em_bits]
        exp_mantissa_b = self.reg_fp_b[0 : self.em_bits]

        # Calculate and register sign
        self.reg_sign.next <<= sign_a ^ sign_b

        # Add the floating point numbers with special lmul offset
        if self._fast:
            final_sum = carrysave_adder(
                exp_mantissa_a,
                exp_mantissa_b,
                self.OFFSET_MINUS_BIAS,
                final_adder=kogge_stone,
            )
        else:
            final_sum = exp_mantissa_a + exp_mantissa_b + self.OFFSET_MINUS_BIAS

        self.reg_final_sum.next <<= final_sum

    def stage2_output_format(self):
        """Format final output"""
        # Mux selection based on overflow/underflow
        mantissa_result = pyrtl.mux(
            self.reg_final_sum[self.em_bits :],  # Select bits for mux control
            pyrtl.Const(0, bitwidth=self.em_bits),  # Underflow case
            self.reg_final_sum[0 : self.em_bits],  # Normal case
            default=self.MAX_VALUE,  # Overflow case
        )

        # Combine sign and mantissa
        self.output_reg.next <<= pyrtl.concat(self.reg_sign, mantissa_result)

    def _build_pipeline(self):
        """Connect all pipeline stages"""
        self.stage0_input()
        self.stage1_split_and_add()
        self.stage2_output_format()
