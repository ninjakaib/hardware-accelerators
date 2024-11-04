from float8 import Float8

def get_lmul_offset(m_bits):
    if m_bits <= 3:
        l = m_bits
    if m_bits == 4:
        l = 3
    if m_bits > 4:
        l = 4
    l = (1 << m_bits) >> l
    return l

def get_exp_bias(e_bits):
    return (2**(e_bits-1) - 1)

def bitmask(n_bits):
    return 2**n_bits - 1

def fp8_lmul_simple(a: Float8, b: Float8) -> Float8:
    # FP8_E4M3 Constants
    e_bits = 4
    m_bits = 3
    total_bits = e_bits + m_bits

    # Get bit representations
    a_bits, b_bits = a.binint, b.binint

    # Step 1: Handle sign
    sign_a = (a_bits >> total_bits) & 1
    sign_b = (b_bits >> total_bits) & 1
    result_sign = sign_a ^ sign_b

    # Step 2: Add the remaining 7 bits (exponent + mantissa)
    fp8_bitmask = bitmask(total_bits)
    remaining_bits_a = a_bits & fp8_bitmask
    remaining_bits_b = b_bits & fp8_bitmask

    l_offset = get_lmul_offset(m_bits)
    exp_bias = get_exp_bias(e_bits) << m_bits

    # Calculate the result with the lmul algorithm
    result = remaining_bits_a + remaining_bits_b + l_offset - exp_bias

    # Clamp to valid 7-bit range
    result = min(max(result, 0), fp8_bitmask)

    # Combine sign and result
    return (result_sign << total_bits) | result