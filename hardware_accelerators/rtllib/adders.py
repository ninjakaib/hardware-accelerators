import pyrtl
from pyrtl import WireVector

from .utils.common import extract_float_components
from .utils.adder_utils import *
from .utils.pipeline import SimplePipeline


### ===================================================================
### Fully Combinatorial Design
### ===================================================================


def float_adder(
    float_a: WireVector,
    float_b: WireVector,
    e_bits: int,
    m_bits: int,
) -> WireVector:
    sign_a, sign_b, exp_a, exp_b, mantissa_a, mantissa_b = extract_float_components(
        float_a, float_b, e_bits, m_bits
    )

    sign_xor, exp_larger, signed_shift, mant_smaller, mant_larger = adder_stage_2(
        sign_a, sign_b, exp_a, exp_b, mantissa_a, mantissa_b, e_bits, m_bits
    )

    abs_shift = WireVector(e_bits)  # , "abs_shift")
    abs_shift <<= signed_shift[:e_bits]

    aligned_mant_msb, sticky_bit, guard_bit, round_bit = adder_stage_3(
        mant_smaller, abs_shift, m_bits, e_bits
    )

    mantissa_sum, is_neg, lzc = adder_stage_4(
        aligned_mant_msb, mant_larger, sign_xor, m_bits
    )

    final_sign, final_exp, norm_mantissa = adder_stage_5(
        mantissa_sum,
        sticky_bit,
        guard_bit,
        round_bit,
        lzc,
        exp_larger,
        sign_a,
        sign_b,
        signed_shift,
        is_neg,
        e_bits,
        m_bits,
    )

    float_result = WireVector(e_bits + m_bits + 1)  # , "float_result")
    float_result <<= pyrtl.concat(final_sign, final_exp, norm_mantissa)
    return float_result


### ===================================================================
### Simple Pipeline Design
### ===================================================================


class FloatAdderPipelined(SimplePipeline):
    def __init__(
        self,
        float_a: WireVector,
        float_b: WireVector,
        w_en: WireVector,
        e_bits: int,
        m_bits: int,
    ):
        """
        Initialize a pipelined BFloat16 adder with write enable control.

        This class implements a 5-stage pipelined BFloat16 addition unit. The stages are:
        1. Input parsing and extraction
        2. Exponent comparison and shift amount calculation
        3. Mantissa alignment and SGR bit generation
        4. Mantissa addition and leading zero detection
        5. Normalization, rounding, and final assembly

        The write enable signal controls when the adder outputs a result. When write_enable
        is low, the output is forced to zero, allowing for selective accumulation when used
        in larger designs.

        Parameters
        ----------
        float_a : WireVector
            First BFloat16 input operand (16 bits)
            Format: [sign(1) | exponent(8) | mantissa(7)]
        float_b : WireVector
            Second BFloat16 input operand (16 bits)
            Format: [sign(1) | exponent(8) | mantissa(7)]
        w_en : WireVector
            Write enable signal (1 bit)
            Controls when the adder outputs a result:
            - 1: Output valid addition result
            - 0: Force output to zero

        Attributes
        ----------
        _result_out : WireVector
            Output wire carrying the addition result (16 bits)
            Format matches BFloat16: [sign(1) | exponent(8) | mantissa(7)]

        Examples
        --------
        >>> # Create input wires
        >>> a = pyrtl.Input(16, 'float_a')
        >>> b = pyrtl.Input(16, 'float_b')
        >>> w_en = pyrtl.Input(1, 'write_enable')
        >>> result = pyrtl.Output(16, 'result')
        >>>
        >>> # Instantiate adder and connect output
        >>> adder = PipelinedBF16Adder2(a, b, w_en)
        >>> result <<= adder._result_out

        Notes
        -----
        - The pipeline has 5 stages with a latency of 5 clock cycles
            (results available after 4 cycles/on the 5th cycle)
        - Write enable should be timed to align with when results should appear
        - Pipeline registers are automatically inserted between stages
        - Uses round-to-nearest-even for rounding

        Raises
        ------
        AssertionError
            If input widths don't match BFloat16 format (16 bits) or
            write enable is not 1 bit
        """
        assert (
            len(float_a) == len(float_b) == 16
        ), f"float inputs must be {e_bits + m_bits + 1} bits"
        assert len(w_en) == 1, "write enable signal must be 1 bit"
        self.e_bits = e_bits
        self.m_bits = m_bits
        # Define inputs and outputs
        self._float_a, self._float_b = float_a, float_b
        self._write_enable = w_en
        # self._result = pyrtl.Register(self.e_bits + self.m_bits + 1, 'result')
        self._result_out = pyrtl.WireVector(e_bits + m_bits + 1)  # , "_result")
        super(FloatAdderPipelined, self).__init__()

    @property
    def result(self):
        return self._result_out

    def stage0(self):
        """Stage 1: Input Parsing and Extraction"""
        # Extract components from inputs
        self.w_en = self._write_enable
        self.sign_a, self.sign_b, self.exp_a, self.exp_b, self.mant_a, self.mant_b = (
            extract_float_components(
                self._float_a, self._float_b, self.e_bits, self.m_bits
            )
        )

    def stage1(self):
        """Stage 2: Exponent Compare and Shift Amount"""
        # Pass through values needed in future stages
        self.w_en = self.w_en
        self.sign_a = self.sign_a
        self.sign_b = self.sign_b

        # Calculate new values
        (
            self.sign_xor,
            self.exp_larger,
            self.signed_shift,
            self.mant_smaller,
            self.mant_larger,
        ) = adder_stage_2(
            self.sign_a,
            self.sign_b,
            self.exp_a,
            self.exp_b,
            self.mant_a,
            self.mant_b,
            self.e_bits,
            self.m_bits,
        )

    def stage2(self):
        """Stage 3: Mantissa Alignment and SGR Generation"""
        # Pass through values needed in future stages
        self.w_en = self.w_en
        self.sign_a = self.sign_a
        self.sign_b = self.sign_b
        self.sign_xor = self.sign_xor
        self.exp_larger = self.exp_larger
        self.signed_shift = self.signed_shift
        self.mant_larger = self.mant_larger

        # Calculate absolute shift amount
        abs_shift = WireVector(self.e_bits)  # , "abs_shift")
        abs_shift <<= self.signed_shift[: self.e_bits]

        # Perform alignment and generate SGR bits
        self.aligned_mant_msb, self.sticky, self.guard, self.round = adder_stage_3(
            self.mant_smaller, abs_shift, self.m_bits, self.e_bits
        )

    def stage3(self):
        """Stage 4: Mantissa Addition and LZD"""
        # Pass through values needed in future stages
        self.w_en = self.w_en
        self.sign_a = self.sign_a
        self.sign_b = self.sign_b
        self.exp_larger = self.exp_larger
        self.signed_shift = self.signed_shift
        self.sticky = self.sticky
        self.guard = self.guard
        self.round = self.round

        # Perform mantissa addition and leading zero detection
        self.mant_sum, self.is_neg, self.lzc = adder_stage_4(
            self.aligned_mant_msb, self.mant_larger, self.sign_xor, self.m_bits
        )

    def stage4(self):
        """Stage 5: Normalization, Rounding, and Final Assembly"""
        # Calculate final values
        final_sign, final_exp, norm_mantissa = adder_stage_5(
            self.mant_sum,
            self.sticky,
            self.guard,
            self.round,
            self.lzc,
            self.exp_larger,
            self.sign_a,
            self.sign_b,
            self.signed_shift,
            self.is_neg,
            self.e_bits,
            self.m_bits,
        )

        # Concatenate final result
        # self._result <<= pyrtl.concat(final_sign, final_exp, norm_mantissa)
        with pyrtl.conditional_assignment:
            with self.w_en:
                self._result_out |= pyrtl.concat(final_sign, final_exp, norm_mantissa)
                # self._result.next |= pyrtl.concat(final_sign, final_exp, norm_mantissa)
            with pyrtl.otherwise:
                # self._result.next |= 0
                self._result_out |= 0


### ===================================================================
