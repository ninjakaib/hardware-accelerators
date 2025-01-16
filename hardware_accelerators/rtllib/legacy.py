import pyrtl
from pyrtl.rtllib.adders import carrysave_adder, kogge_stone
from hardware_accelerators.rtllib.utils.lmul_utils import get_combined_offset


###########################
# Old code below
###########################


# BF16 Naive Combinatorial
def bf16_lmul_naive():
    # inputs and outputs
    fp_a = pyrtl.Input(bitwidth=16, name="fp_a")
    fp_b = pyrtl.Input(bitwidth=16, name="fp_b")
    fp_out = pyrtl.Output(bitwidth=16, name="fp_out")

    sign_out = fp_a[15] ^ fp_b[15]

    unsigned_offset = pyrtl.Const(get_combined_offset(8, 7), 15)
    result_sum = fp_a[:15] + fp_b[:15] - unsigned_offset
    fp_out <<= pyrtl.concat(sign_out, pyrtl.truncate(result_sum, 15))


# bfloat16 Fast Combinatorial
@pyrtl.wire_struct
class BF16Wire:
    sign: 1
    exp_mant: 15


@pyrtl.wire_struct
class BF16_LMUL_SUM:
    carry_mux: 2
    sum_bits: 15


def bf16_lmul_combinatorial():
    # Inputs
    em_bits = 15
    fp_a = BF16Wire(name="fp_a", concatenated_type=pyrtl.Input)
    fp_b = BF16Wire(name="fp_b", concatenated_type=pyrtl.Input)
    # expected_out = BF16Wire(name='expected_out', concatenated_type=pyrtl.Input)

    # Calculate result sign
    sign_output = pyrtl.WireVector(1, name="sign_output")
    sign_output <<= fp_a.sign ^ fp_b.sign

    OFFSET_MINUS_BIAS = pyrtl.Const(
        get_combined_offset(8, 7, twos_comp=True), bitwidth=em_bits, name="offsetbias"
    )

    final_sum = BF16_LMUL_SUM(
        name="final_sum",
        BF16_LMUL_SUM=carrysave_adder(
            fp_a.exp_mant, fp_b.exp_mant, OFFSET_MINUS_BIAS, final_adder=kogge_stone
        ),
    )

    MIN_VALUE = pyrtl.Const(0, bitwidth=em_bits, name="min_value")
    MAX_VALUE = pyrtl.Const(2**em_bits - 1, bitwidth=em_bits, name="max_value")

    exp_mant = pyrtl.WireVector(15, name="exp_mant_mux")
    exp_mant <<= pyrtl.mux(
        final_sum.carry_mux,
        MIN_VALUE,
        final_sum.sum_bits,
        default=MAX_VALUE,
    )

    fp_out = BF16Wire(
        name="fp_out",
        sign=sign_output,
        exp_mant=exp_mant,
        concatenated_type=pyrtl.Output,
    )

    return fp_a, fp_b, fp_out


# float8 fast lmul
def fp8_lmul_combinatorial():
    # Inputs
    fp_a = pyrtl.Input(8, "fp_a")
    fp_b = pyrtl.Input(8, "fp_b")
    fp_out = pyrtl.Output(8, "fp_out")

    # Split into sign and exp_mantissa parts
    sign_a = fp_a[7]
    sign_b = fp_b[7]
    exp_mantissa_a = fp_a[0:7]
    exp_mantissa_b = fp_b[0:7]

    # Calculate result sign
    result_sign = sign_a ^ sign_b

    # Add exp_mantissa parts using kogge_stone adder (faster than ripple)
    # exp_mantissa_sum = kogge_stone(exp_mantissa_a, exp_mantissa_b)

    # For E4M3: e_bits=4, m_bits=3
    # Get the combined offset-bias constant
    OFFSET_MINUS_BIAS = pyrtl.Const(get_combined_offset(4, 3, True), bitwidth=7)

    # Add offset-bias value - this will be 8 bits including carry
    # final_sum = kogge_stone(exp_mantissa_sum, OFFSET_MINUS_BIAS)

    final_sum = carrysave_adder(
        exp_mantissa_a, exp_mantissa_b, OFFSET_MINUS_BIAS, final_adder=kogge_stone
    )
    # Extract carry and MSB for overflow/underflow detection
    # mux_in = final_sum[7:]  # 8th and 9th bits
    # result_bits = final_sum[0:7]  # lower 7 bits

    # Select result based on carry and MSB:
    # carry=1: overflow -> 0x7F
    # carry=0, msb=0: underflow -> 0x00
    # carry=0, msb=1: normal -> result_bits
    MAX_VALUE = pyrtl.Const(0x7F, 7)

    mantissa_result = pyrtl.mux(
        final_sum[7:], pyrtl.Const(0, bitwidth=7), final_sum[0:7], default=MAX_VALUE
    )

    # Combine sign and result
    fp_out <<= pyrtl.concat(result_sign, mantissa_result)

    return fp_a, fp_b, fp_out


# Float8 fast pipelined lmul
class FastPipelinedLMULFP8:
    def __init__(self):
        # Inputs and Outputs
        self.fp_a = pyrtl.Input(8, "fp_a")
        self.fp_b = pyrtl.Input(8, "fp_b")
        self.fp_out = pyrtl.Output(8, "fp_out")

        # Constants
        self.OFFSET_MINUS_BIAS = pyrtl.Const(
            get_combined_offset(4, 3, twos_comp=True), bitwidth=7
        )
        self.MAX_VALUE = pyrtl.Const(0x7F, 7)

        # Pipeline Registers
        # Stage 0 -> 1
        self.reg_fp_a = pyrtl.Register(8, "reg_fp_a")
        self.reg_fp_b = pyrtl.Register(8, "reg_fp_b")

        # Stage 1 -> 2
        self.reg_sign = pyrtl.Register(1, "reg_sign")
        self.reg_final_sum = pyrtl.Register(9, "reg_final_sum")

        # Stage 2 -> output
        self.reg_output = pyrtl.Register(8, "reg_output")

        # Build pipeline
        self._build_pipeline()

    def stage0_input(self):
        """Input registration stage"""
        self.reg_fp_a.next <<= self.fp_a
        self.reg_fp_b.next <<= self.fp_b

    def stage1_split_and_add(self):
        """Split inputs and perform additions"""
        # Split registered inputs
        sign_a = self.reg_fp_a[7]
        sign_b = self.reg_fp_b[7]
        exp_mantissa_a = self.reg_fp_a[0:7]
        exp_mantissa_b = self.reg_fp_b[0:7]

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
            self.reg_final_sum[7:],  # Select bits for mux control
            pyrtl.Const(0, bitwidth=7),  # Underflow case
            self.reg_final_sum[0:7],  # Normal case
            default=self.MAX_VALUE,  # Overflow case
        )

        # Combine sign and mantissa
        self.reg_output.next <<= pyrtl.concat(self.reg_sign, mantissa_result)

    def _build_pipeline(self):
        """Connect all pipeline stages"""
        self.stage0_input()
        self.stage1_split_and_add()
        self.stage2_output_format()
        # Connect final register to output
        self.fp_out <<= self.reg_output
