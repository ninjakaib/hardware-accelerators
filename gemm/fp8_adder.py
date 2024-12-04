from pyrtl import *
from pyrtl.corecircuits import shift_right_arithmetic, shift_right_logical

def fp8e4m3_adder(a, b):
    """
    FP8 (e4m3) adder implementation
    Format: [sign(1) | exponent(4) | mantissa(3)]
    """
    # Extract components
    a_sign = a[7]
    b_sign = b[7]
    a_exp = a[3:7]
    b_exp = b[3:7]
    # Add implicit 1 to mantissa
    a_mant = concat(Const(1, 1), a[0:3])  
    b_mant = concat(Const(1, 1), b[0:3])
    
    # Calculate exponent difference
    exp_diff = a_exp - b_exp
    
    # Determine which number is larger in magnitude
    a_larger = (a_exp > b_exp) | ((a_exp == b_exp) & (a_mant >= b_mant))
    
    # Align mantissas based on exponent difference
    larger_exp = select(a_larger, a_exp, b_exp)
    larger_mant = select(a_larger, a_mant, b_mant)
    larger_sign = select(a_larger, a_sign, b_sign)
    smaller_mant = select(a_larger, b_mant, a_mant)
    smaller_sign = select(a_larger, b_sign, a_sign)
    
    # Shift smaller mantissa right based on exponent difference
    # Use logical right shift since these are unsigned mantissas
    shift_amount = select(exp_diff > 8, Const(8, 4), exp_diff[0:4])
    shifted_mant = shift_right_logical(smaller_mant, shift_amount)
    
    # Add or subtract mantissas based on signs
    add_sub = larger_sign ^ smaller_sign
    aligned_larger = concat(Const(0, 1), larger_mant)  # Add leading 0 for overflow
    aligned_smaller = concat(Const(0, 1), shifted_mant)
    
    raw_mant = select(add_sub,
                     aligned_larger - aligned_smaller,
                     aligned_larger + aligned_smaller)
    
    # Normalize result
    # Find leading 1 position
    has_leading_1 = raw_mant[4]
    has_leading_0 = ~raw_mant[4] & raw_mant[3]
    
    # Adjust mantissa and exponent based on normalization
    final_mant = select(has_leading_1,
                       raw_mant[1:4],  # Shift right case
                       select(has_leading_0,
                             raw_mant[0:3],  # No shift case
                             concat(raw_mant[2:4], Const(0, 1))))  # Shift left case
    
    exp_adjust = select(has_leading_1,
                       Const(1, 4),  # Add 1 to exponent
                       select(has_leading_0,
                             Const(0, 4),  # No change
                             Const(-1, 4)))  # Subtract 1 from exponent
    
    final_exp = larger_exp + exp_adjust
    
    # Handle special cases
    is_zero = (raw_mant == 0)
    
    # Detect overflow/underflow
    overflow = (final_exp >= 15)  # Max exponent value for e4
    underflow = (final_exp < 0)
    
    # Compose final result
    result = select(is_zero,
                   Const(0, 8),
                   concat_list([
                       larger_sign,  # Sign
                       select(overflow, Const(0b1111, 4), final_exp),  # Exponent
                       final_mant  # Mantissa
                   ]))
    
    return result, overflow, underflow

def test_fp8e4m3_adder():
    # Create inputs
    a = Input(8, 'a')
    b = Input(8, 'b')
    
    # Connect to adder
    result, overflow, underflow = fp8e4m3_adder(a, b)
    
    # Create outputs
    sum_out = Output(8, 'sum')
    overflow_out = Output(1, 'overflow')
    underflow_out = Output(1, 'underflow')
    
    sum_out <<= result
    overflow_out <<= overflow
    underflow_out <<= underflow
    
    # Create simulation
    sim = Simulation()
    
    # Test cases
    test_cases = [
        # Format: (a, b, expected_sum, expected_overflow, expected_underflow)
        # Simple positive additions
        (0b00111000, 0b0111000, 0b01000000, 0, 0),  # 1.0 + 1.0 = 2.0
        (0b00010100, 0b00010000, 0b00100010, 0, 0),  # 1.5 + 1.0 = 2.5
        
        # Different exponents
        (0b00100000, 0b00010000, 0b00100100, 0, 0),  # 2.0 + 1.0 = 3.0
        
        # Negative numbers
        (0b10010000, 0b00010000, 0b00000000, 0, 0),  # -1.0 + 1.0 = 0.0
        (0b10010000, 0b10010000, 0b10100000, 0, 0),  # -1.0 + (-1.0) = -2.0
        
        # Overflow cases
        (0b01110000, 0b01110000, 0b01111000, 1, 0),  # Large + Large = Overflow
        
        # Underflow cases
        (0b00000100, 0b10000100, 0b00000000, 0, 1),  # Small - Small = Underflow
        
        # Zero cases
        (0b00000000, 0b00010000, 0b00010000, 0, 0),  # 0.0 + 1.0 = 1.0
        (0b00000000, 0b00000000, 0b00000000, 0, 0),  # 0.0 + 0.0 = 0.0
    ]
    
    # Run test cases
    for i, (a_val, b_val, expected_sum, expected_overflow, expected_underflow) in enumerate(test_cases):
        sim.step({
            'a': a_val,
            'b': b_val
        })
        
        actual_sum = sim.inspect('sum')
        actual_overflow = sim.inspect('overflow')
        actual_underflow = sim.inspect('underflow')
        
        # Print results
        print(f"\nTest case {i}:")
        print(f"a:        {bin(a_val)[2:].zfill(8)}")
        print(f"b:        {bin(b_val)[2:].zfill(8)}")
        print(f"Expected: {bin(expected_sum)[2:].zfill(8)}")
        print(f"Got:      {bin(actual_sum)[2:].zfill(8)}")
        print(f"Overflow: expected={expected_overflow}, got={actual_overflow}")
        print(f"Underflow: expected={expected_underflow}, got={actual_underflow}")
        
        # Assert correctness
        assert actual_sum == expected_sum, f"Sum mismatch in test case {i}"
        assert actual_overflow == expected_overflow, f"Overflow mismatch in test case {i}"
        assert actual_underflow == expected_underflow, f"Underflow mismatch in test case {i}"
    
    print("\nAll tests passed!")

# Run the tests
if __name__ == "__main__":
    test_fp8e4m3_adder()
