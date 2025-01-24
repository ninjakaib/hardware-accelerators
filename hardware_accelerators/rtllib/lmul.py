from typing import Type
import pyrtl
from pyrtl import WireVector
from pyrtl.rtllib.adders import carrysave_adder, kogge_stone

from .utils.lmul_utils import get_combined_offset
from ..dtypes import BaseFloat, Float8


def lmul_simple(
    float_a: WireVector,
    float_b: WireVector,
    dtype: Type[BaseFloat],
):
    """Linear time complexity float multiply unit in the simplest configuration."""
    e_bits, m_bits = dtype.exponent_bits(), dtype.mantissa_bits()
    em_bits = e_bits + m_bits
    sign_out = float_a[em_bits] ^ float_b[em_bits]

    unsigned_offset = pyrtl.Const(get_combined_offset(e_bits, m_bits), em_bits)
    result_sum = float_a[:em_bits] + float_b[:em_bits] - unsigned_offset

    fp_out = WireVector(bitwidth=em_bits + 1)
    fp_out <<= pyrtl.concat(sign_out, pyrtl.truncate(result_sum, em_bits))
    return fp_out


def lmul_fast(float_a: WireVector, float_b: WireVector, dtype: Type[BaseFloat]):
    e_bits, m_bits = dtype.exponent_bits(), dtype.mantissa_bits()
    em_bits = e_bits + m_bits
    sign_a = float_a[em_bits]
    sign_b = float_b[em_bits]
    exp_mantissa_a = float_a[:em_bits]
    exp_mantissa_b = float_b[:em_bits]
    fp_out = WireVector(em_bits + 1)

    # Calculate result sign
    result_sign = sign_a ^ sign_b

    # Add exp_mantissa parts using kogge_stone adder (faster than ripple)
    # exp_mantissa_sum = kogge_stone(exp_mantissa_a, exp_mantissa_b)

    # Get the combined offset-bias constant
    OFFSET_MINUS_BIAS = pyrtl.Const(
        get_combined_offset(e_bits, m_bits, True), bitwidth=em_bits
    )

    # Add offset-bias value - this will be 8 bits including carry
    # final_sum = kogge_stone(exp_mantissa_sum, OFFSET_MINUS_BIAS)

    final_sum = carrysave_adder(
        exp_mantissa_a, exp_mantissa_b, OFFSET_MINUS_BIAS, final_adder=kogge_stone
    )

    # Select result based on carry and MSB:
    # carry=1: overflow -> 0x7F
    # carry=0, msb=0: underflow -> 0x00
    # carry=0, msb=1: normal -> result_bits

    MAX_VALUE = pyrtl.Const(2**em_bits - 1, bitwidth=em_bits)  # , name="max_value")

    if e_bits == 4 and m_bits == 3:
        MAX_VALUE = pyrtl.Const(0x7F, 7)

    mantissa_result = pyrtl.mux(
        final_sum[em_bits:],
        pyrtl.Const(0, bitwidth=em_bits),
        final_sum[:em_bits],
        default=MAX_VALUE,
    )

    # Combine sign and result
    fp_out <<= pyrtl.concat(result_sign, mantissa_result)

    return fp_out


# Float8 fast pipelined lmul
class LmulPipelined:
    def __init__(
        self,
        float_a: WireVector,
        float_b: WireVector,
        dtype: Type[BaseFloat],
    ):
        self.e_bits = dtype.exponent_bits()
        self.m_bits = dtype.mantissa_bits()
        self.em_bits = dtype.bitwidth() - 1

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

        # First addition and register result
        final_sum = carrysave_adder(
            exp_mantissa_a,
            exp_mantissa_b,
            self.OFFSET_MINUS_BIAS,
            final_adder=kogge_stone,
        )

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
