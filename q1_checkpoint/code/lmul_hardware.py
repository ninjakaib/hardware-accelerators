import pyrtl
from pyrtl.rtllib.libutils import twos_comp_repr
from pyrtl.rtllib.adders import carrysave_adder, kogge_stone
def get_const_offset(e_bits, m_bits):
    total_bits = e_bits + m_bits
    bias = (2**(e_bits-1) - 1) << m_bits
    
    if m_bits <= 3:
        l = m_bits
    if m_bits == 4:
        l = 3
    if m_bits > 4:
        l = 4
    l = (1 << m_bits) >> l
    
    offset = twos_comp_repr(l-bias, total_bits)
    return offset

def lmul_rtl():
    # Inputs 
    fp_a = pyrtl.Input(8, 'fp_a')
    fp_b = pyrtl.Input(8, 'fp_b')
    fp_out = pyrtl.Output(8, 'fp_out')

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
    OFFSET_MINUS_BIAS = pyrtl.Const(get_const_offset(4, 3), bitwidth=7)
    
    # Add offset-bias value - this will be 8 bits including carry
    # final_sum = kogge_stone(exp_mantissa_sum, OFFSET_MINUS_BIAS)
    
    final_sum = carrysave_adder(exp_mantissa_a, exp_mantissa_b, OFFSET_MINUS_BIAS, final_adder=kogge_stone)
    
    # Extract carry and MSB for overflow/underflow detection
    carry = final_sum[8]  # 9th bit
    msb = final_sum[7]    # 8th bit
    result_bits = final_sum[0:7]  # lower 7 bits

    # Select result based on carry and MSB:
    # carry=1: overflow -> 0x7F
    # carry=0, msb=0: underflow -> 0x00
    # carry=0, msb=1: normal -> result_bits
    MAX_VALUE = pyrtl.Const(0x7F, 7)
    
    with pyrtl.conditional_assignment:
        with carry:
            mantissa_result = MAX_VALUE
        with ~carry & ~msb:
            mantissa_result = 0
        with ~carry & msb:
            mantissa_result = result_bits

    # Combine sign and result
    fp_out <<= pyrtl.concat(result_sign, mantissa_result)

    return fp_a, fp_b, fp_out
