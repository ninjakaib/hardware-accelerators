from pyrtl.rtllib.libutils import twos_comp_repr

def get_shifted_bias(e_bits, m_bits):
    return (2**(e_bits-1) - 1) << m_bits

def get_lmul_lk_offset(m_bits):
    if m_bits <= 3:
        l = m_bits
    if m_bits == 4:
        l = 3
    if m_bits > 4:
        l = 4
    l = (1 << m_bits) >> l 
    return l

def get_combined_offset(e_bits, m_bits, twos_comp=False, fmt:str=None):
    """
    Calculate the offset term for the LMUL (Large Multiply) algorithm for a given floating-point format.

    This function computes the offset needed to perform the LMUL algorithm by:
    1. Calculating the exponent bias shifted to align with the exponent bits
    2. Determining the LMUL L(k) offset term, where k is the number of mantissa bits
    3. Subtracting the L(k) offset from the shifted bias

    The returned value should be subtracted from the sum of two input floats.

    Args:
        e_bits (int): Number of exponent bits in the format
        m_bits (int): Number of mantissa bits in the format
        twos_comp (bool, optional): If True, returns the offset in two's complement representation. 
            Defaults to False.
        fmt (str, optional): Format specifier for output representation. 
            If starts with 'b', returns binary with bit sections separated by '_'. 
            Otherwise, formats the offset according to the specified format. Defaults to None.

    Returns:
        int or str: The calculated offset value, either as an integer or formatted string 
        depending on the `fmt` parameter.
    """
    total_bits = e_bits + m_bits
    bias = get_shifted_bias(e_bits, m_bits)
    l = get_lmul_lk_offset(m_bits)
    offset = bias - l
    if twos_comp:
        offset = twos_comp_repr(-offset, total_bits)
    if fmt:
        if fmt.startswith('b'):
            formatted_offset = format(offset, f"0{total_bits+1}{fmt[0]}")
            return formatted_offset[0] + '_' + formatted_offset[1:e_bits+1] + '_' + formatted_offset[e_bits+1:]
        return format(offset, f"0{fmt[0]}")
    return offset

OFFSETS = {
    "fp8e4m3": get_combined_offset(4, 3, twos_comp=False),
    "fp8e4m3_twos_comp": get_combined_offset(4, 3, twos_comp=True),
    "fp8e4m3_bin": get_combined_offset(4, 3, twos_comp=False, fmt='b'),
    "fp8e4m3_twos_comp_bin": get_combined_offset(4, 3, twos_comp=True, fmt='b'),
    "bf_16": get_combined_offset(8, 7, twos_comp=False),
    "bf_16_twos_comp": get_combined_offset(8, 7, twos_comp=True),
    "bf16_bin": get_combined_offset(8, 7, twos_comp=False, fmt='b'),
    "bf16_twos_comp_bin": get_combined_offset(8, 7, twos_comp=True, fmt='b'),
    "fp32": get_combined_offset(8, 23, twos_comp=False),
    "fp32_twos_comp": get_combined_offset(8, 23, twos_comp=True),
    "fp32_bin": get_combined_offset(8, 23, twos_comp=False, fmt='b'),
    "fp32_twos_comp_bin": get_combined_offset(8, 23, twos_comp=True, fmt='b'),
}
