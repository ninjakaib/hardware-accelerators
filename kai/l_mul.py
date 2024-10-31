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
