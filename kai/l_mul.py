from float8 import Float8

def lmul(x: Float8, y: Float8):
    x_bits, y_bits = int(x.bits(), 2), int(y.bits(), 2)
    binary_result = binary_fp8e4m3_lmul(x_bits, y_bits)
    binary_result = Float8(format(binary_result, '08b'))
    true_result = Float8(x.decimal * y.decimal)
    return {
        'binary_result': binary_result,
        'true_result': true_result,
        'error': abs(true_result.decimal - binary_result.decimal)
    }


def binary_fp8e4m3_lmul(x: int, y: int) -> int:
    # Step 1: Handle sign bit 
    sign_x = (x >> 7) & 1
    sign_y = (y >> 7) & 1
    result_sign = sign_x ^ sign_y
    
    # Step 2: Add the remaining 7 bits (exponent + mantissa)
    remaining_bits_x = x & 0x7F
    remaining_bits_y = y & 0x7F
    
    # For e4m3 format:
    # - l(m) = 3 since m = 3
    # - 2^(-l(m)) = 2^(-3) = 0.125 in normalized form
    # - In our fixed point representation, this is adding 1 to mantissa bits
    # - Bias adjustment needs -7 since we're adding exponents
    result = remaining_bits_x + remaining_bits_y + (1 << 0) - (7 << 3)
    # display_float8_conversion(binary(result, 8))
    # return result
    # Clamp to valid 7-bit range
    result = min(max(result, 0), 0x7F)
    
    # Combine sign and result
    return (result_sign << 7) | result


# # Input: x = 0b1001100 (6.0), y = 0b0111100 (1.5)
# x, y = 0b00111000, 0b01000000
# result = float8_lmul(x, y)
# print(f"Result (binary): {result:08b}")
# display_float8_conversion(f"{result:08b}")
def get_shift_amount(mantissa: int, m_bits: int) -> int:
    # If mantissa is 0, no shift will make it normalized
    if mantissa == 0:
        return 0
        
    # Find position of leftmost 1 by checking each bit
    # from left to right
    for i in range(m_bits):
        if mantissa & (1 << (m_bits - i)):
            # Return number of shifts needed to get leftmost 1
            # to implied position
            return i
            
    return m_bits # In case no 1 is found

print(get_shift_amount(0b010, 3)) # Prints 2
print(get_shift_amount(0b0010, 4)) # Prints 3
print(get_shift_amount(0b101, 3)) # Prints 1

def l_offset(m_bits):
    if m_bits <= 3:
        offset = m_bits
    if m_bits == 4:
        offset = 3
    if m_bits > 4:
        offset = 4
    offset = (1 << m_bits) >> offset
    # print(format(offset, f'0{m_bits}b'))
    return offset
    
l_offset(3)
def lmul_normsub(x:float|str, y:float|str):
    # Constants for e4m3 format
    E_BITS = 4
    M_BITS = 3
    PRECISION = 1 + E_BITS + M_BITS
    BIAS = 7  # 2^(4-1) - 1
    EXP_MASK = (1 << E_BITS) - 1
    MANTISSA_MASK = (1 << M_BITS) - 1
    MAX_EXP = (1 << E_BITS) - 1  # 15
    MAX_VALUE = 0x7F  # Maximum 7-bit value (excluding sign)
    
    xfloat8 = Float8(x)
    yfloat8 = Float8(y)
    
    print(f"x = {xfloat8}")
    print(f"y = {yfloat8}")
    
    # Step 0: Check for subnormal inputs
    x_exp = (xfloat8.binint >> M_BITS) & EXP_MASK
    y_exp = (yfloat8.binint >> M_BITS) & EXP_MASK
    x_mantissa = xfloat8.binint & MANTISSA_MASK
    y_mantissa = yfloat8.binint & MANTISSA_MASK
    
    # Step 1: Handle sign bit
    sign_x = (xfloat8.binint >> 7) & 1
    sign_y = (yfloat8.binint >> 7) & 1
    result_sign = sign_x ^ sign_y
    
    if x_exp == 0 or y_exp == 0:
        # Get the subnormal number's mantissa and exp
        subnormal_mantissa = x_mantissa if x_exp == 0 else y_mantissa
        normal_mantissa = y_mantissa if x_exp == 0 else x_mantissa
        normal_exp = y_exp if x_exp == 0 else x_exp
        
        # # Strategy 1: Add the special offset term after normalizing the mantissa
        # normalizing_shift = get_shift_amount(subnormal_mantissa, M_BITS)
        # print("Normalized shift amount: ", normalizing_shift)
        # result_mantissa = (subnormal_mantissa << normalizing_shift) & MANTISSA_MASK
        # result_mantissa += l_offset(M_BITS)
        
        # Strategy 2: Add the special offset term before normalization
        result_mantissa = subnormal_mantissa + l_offset(M_BITS)
        normalizing_shift = get_shift_amount(result_mantissa, M_BITS)
        print("Normalized shift amount: ", normalizing_shift)
        result_mantissa = (result_mantissa << normalizing_shift) & MANTISSA_MASK
        result_mantissa += normal_mantissa
        mantissa_carry = (result_mantissa >> M_BITS)
        result_mantissa = result_mantissa & MANTISSA_MASK
        print(format(result_mantissa, '0b'))
        print(f"Normalized shift: {normalizing_shift}\nNormal exp: {normal_exp}\nBIAS: {BIAS}\ncarry: {mantissa_carry}")
        
        # The result's exponent is the non-zero exponent of the normal number
        result_exp = normal_exp + 1 - BIAS - normalizing_shift + mantissa_carry
        
    # If the exponent <= 0, denormalize the result
    if result_exp <= 0:
        print("result exp:", result_exp)
        print("normalized mantissa result: 1."+format(result_mantissa, f'0{M_BITS}b'))
        denormalizing_shift = abs(result_exp) + 1
        result_exp = 0
        result_mantissa = ((1 << M_BITS) + result_mantissa) >> (denormalizing_shift)
        print("denormalized mantissa result: 0."+format(result_mantissa, f'0{M_BITS}b'))    
    # Combine results
    result = (result_sign << PRECISION-1) | (result_exp << M_BITS) | result_mantissa
    print(format(result, '08b'))
    result = Float8.from_binint(result)
    print(f"True result:\n\t{xfloat8*yfloat8}")
    print(f"Calculated result:\n\t{result}\n")
    return 

# lmul_binary('0.1110.000', '0.0000.111')
# lmul_normsub('0.1110.000', '0.0000.111')
x, y = 1, 0.005
lmul_normsub(x,y)