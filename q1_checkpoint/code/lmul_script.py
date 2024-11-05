from float8 import Float8
from lmul import fp8_lmul_simple
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations, product, combinations_with_replacement

def generate_normal_fp8_list():
    fp8_list = [Float8(0)]
    for s in range(2):
        for e in range(1,16):
            for m in range(8):
                sign = format(s, '01b')
                exponent = format(e, '04b')
                mantissa = format(m, '03b')
                fp8_list.append(Float8(sign + exponent + mantissa))
        # When all bits are 1, it represents inf, we can exclude these
        fp8_list.pop(-1)
    return fp8_list

def generate_all_normal_combinations():
    fp8_list = generate_normal_fp8_list()
    # Generate all combinations of fp8_list with itself
    return list(combinations_with_replacement(fp8_list, 2))

def generate_combinations_for_heatmap():
    fp8_list = [Float8(0)]
    for s in range(2):
        for e in range(1,16):
            for m in [0,7]:
                sign = format(s, '01b')
                exponent = format(e, '04b')
                mantissa = format(m, '03b')
                fp8_list.append(Float8(sign + exponent + mantissa))
    return list(combinations_with_replacement(fp8_list, 2))

def test_combinations(test_cases: list[tuple[Float8, Float8]], multiply_func):
    # Test all possible combinations of subnormal and normal numbers
    # test_cases = generate_all_normal_combinations()
    # test_cases = generate_combinations_for_heatmap()
    results = []
    for x, y in test_cases:
            true_result = x * y
            calculated_result = multiply_func(x, y)
            results.append({
                "x_binary": x.binary,
                "y_binary": y.binary,
                "x_decimal": x.decimal,
                "y_decimal": y.decimal,
                "true_result_binary": true_result.binary,
                "true_result": true_result.decimal,
                "true_result_decimal_approx": true_result.decimal_approx,
                "calculated_result": calculated_result.decimal,
                "calculated_result_binary": calculated_result.binary,
                # We calculate the error using the decimal approximations from the binary representations
                # since by nature of the FP8 format it cannot accurately represent all decimal numbers
                "error": calculated_result.decimal_approx - true_result.decimal_approx,
                "true_total_error": calculated_result.decimal - true_result.decimal,
            })

    df = pd.DataFrame(results)
    return df

all_normals = generate_all_normal_combinations()
subset_normals = generate_combinations_for_heatmap()

df_normals = test_combinations(all_normals, fp8_lmul_simple)

def plot_multiplication_error_heatmap(df, filename):
    # Create matrices for x and y values to plot
    unique_values = sorted(list(set(df['x_decimal'].unique()) | set(df['y_decimal'].unique())))
    n = len(unique_values)
    error_matrix = np.zeros((n, n))
    
    # Fill the error matrix
    value_to_index = {val: idx for idx, val in enumerate(unique_values)}
    for _, row in df.iterrows():
        i = value_to_index[row['x_decimal']]
        j = value_to_index[row['y_decimal']]
        error_matrix[i, j] = row['error']
        # Make the matrix symmetric since multiplication is commutative
        error_matrix[j, i] = row['error']
    
    # Create the heatmap
    plt.figure(figsize=(8, 6))
    
    # Use diverging colormap with white at center
    max_abs_error = np.max(np.abs(error_matrix))
    heatmap = plt.imshow(error_matrix, 
                        cmap='RdBu_r',  # Red-White-Blue colormap
                        aspect='equal',
                        vmin=-max_abs_error,
                        vmax=max_abs_error)
    
    # Add colorbar
    plt.colorbar(heatmap, label='Error Magnitude')
    
    # Customize the plot
    plt.title('FP8 Multiplication Error Heatmap')
    plt.xlabel('Second Operand Value')
    plt.ylabel('First Operand Value')
    
    # Set tick labels (use fewer ticks for readability)
    tick_indices = np.linspace(0, n-1, min(10, n)).astype(int)
    plt.xticks(tick_indices, [f'{unique_values[i]:.2f}' for i in tick_indices], rotation=45)
    plt.yticks(tick_indices, [f'{unique_values[i]:.2f}' for i in tick_indices])
    
    plt.tight_layout()
    plt.savefig(filename)

plot_multiplication_error_heatmap(df_normals, 'output/fp8_multiplication_error.png')

from lmul import get_lmul_offset

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

def lmul_normal_subnormal(x:Float8, y:Float8) -> Float8:
    # Constants for e4m3 format
    E_BITS = 4
    M_BITS = 3
    PRECISION = 1 + E_BITS + M_BITS
    BIAS = 7  # 2^(4-1) - 1
    EXP_MASK = (1 << E_BITS) - 1
    MANTISSA_MASK = (1 << M_BITS) - 1
        
    # Step 0: Check for subnormal inputs
    x_exp = (x.binint >> M_BITS) & EXP_MASK
    y_exp = (y.binint >> M_BITS) & EXP_MASK
    x_mantissa = x.binint & MANTISSA_MASK
    y_mantissa = y.binint & MANTISSA_MASK
    
    # Step 1: Handle sign bit
    sign_x = (x.binint >> 7) & 1
    sign_y = (y.binint >> 7) & 1
    result_sign = sign_x ^ sign_y
    
    if x_exp == 0 or y_exp == 0:
        # 0. Identify subnormal and normal numbers
        subnormal_mantissa = x_mantissa if x_exp == 0 else y_mantissa
        normal_mantissa = y_mantissa if x_exp == 0 else x_mantissa
        normal_exp = y_exp if x_exp == 0 else x_exp
        
        # First normalize the subnormal mantissa
        subnormal_mantissa += get_lmul_offset(M_BITS)
        norm_shift = get_shift_amount(subnormal_mantissa, M_BITS)
        normalized_mantissa = (subnormal_mantissa << norm_shift) & MANTISSA_MASK
        
        # Calculate the result mantissa from normalized values
        result_mantissa = normal_mantissa + normalized_mantissa #+ l_offset(M_BITS)
        
        # Check if the mantissa addition step resulted in an overflow
        if result_mantissa > MANTISSA_MASK:
            result_mantissa &= MANTISSA_MASK
            result_mantissa >>= 1
            norm_shift += 1
        
         # 2. Adjusted exponent for normalized subnormal
        subnorm_adj_exp = (1 - BIAS) - norm_shift
        
        # 3. Calculate the result exponent
        result_exp = normal_exp + subnorm_adj_exp
        
        # Check for 0 exponent after addition, which means the result is denormalized
        if result_exp <= 0:
            # Denormalize result
            denorm_shift = 1 - result_exp
            result_mantissa = ((1<<M_BITS)+result_mantissa) >> denorm_shift
            result_exp = 0

    # Combine results
    result = (result_sign << PRECISION-1) | (result_exp << M_BITS) | result_mantissa

    return Float8.from_binint(result)

def generate_all_subnormals():
    subnormals = []
    for s in range(2):
        for m in range(1,8):
            sign = format(s, '01b')
            exponent = '0000'
            mantissa = format(m, '03b')
            subnormals.append(Float8(sign + exponent + mantissa))
    return subnormals

def generate_normal_subnormal_combinations():
    subnormals = generate_all_subnormals()
    normals = generate_normal_fp8_list()
    return list(product(normals, subnormals))

normal_subnormals = generate_normal_subnormal_combinations()

def plot_normal_subnormal_error_heatmap(df, filename):
    # Get unique normal and subnormal values
    normal_values = sorted(df['x_decimal'].unique())
    subnormal_values = sorted(df['y_decimal'].unique())
    
    # Create the error matrix
    error_matrix = np.zeros((len(subnormal_values), len(normal_values)))
    
    # Create mapping dictionaries for quick index lookup
    normal_to_idx = {val: idx for idx, val in enumerate(normal_values)}
    subnormal_to_idx = {val: idx for idx, val in enumerate(subnormal_values)}
    
    # Fill the error matrix
    for _, row in df.iterrows():
        i = subnormal_to_idx[row['y_decimal']]  # subnormal on y-axis
        j = normal_to_idx[row['x_decimal']]     # normal on x-axis
        error_matrix[i, j] = row['error']
    
    # Create the plot
    plt.figure(figsize=(15, 6))
    
    # Get true min and max errors
    max_error = df['error'].max()
    min_error = df['error'].min()
    abs_max = max(abs(max_error), abs(min_error))
    
    # Create heatmap
    heatmap = plt.imshow(error_matrix, 
                        cmap='RdBu_r',
                        aspect='auto',  # Changed from 'equal' to 'auto' for non-square matrix
                        vmin=-abs_max,
                        vmax=abs_max)
    
    # Add colorbar
    plt.colorbar(heatmap, label='Error Magnitude')
    
    # Customize the plot
    plt.title('Normal × Subnormal FP8 Multiplication Error Heatmap')
    plt.xlabel('Normal Number Value')
    plt.ylabel('Subnormal Number Value')
    
    # Set tick labels (use fewer ticks for readability)
    # For x-axis (normal values)
    n_normal_ticks = min(10, len(normal_values))
    normal_tick_indices = np.linspace(0, len(normal_values)-1, n_normal_ticks).astype(int)
    plt.xticks(normal_tick_indices, 
               [f'{normal_values[i]:.2f}' for i in normal_tick_indices], 
               rotation=45)
    
    # For y-axis (subnormal values), show all since there are only 14
    plt.yticks(range(len(subnormal_values)), 
               [f'{val:.6f}' for val in subnormal_values])
    
    plt.tight_layout()
    plt.savefig(filename)

df_subnormals_naiive = test_combinations(normal_subnormals, fp8_lmul_simple)
plot_normal_subnormal_error_heatmap(df_subnormals_naiive, 'output/basic_lmul_normalxsubnormal.png')

df_subnormals = test_combinations(normal_subnormals, lmul_normal_subnormal)
plot_normal_subnormal_error_heatmap(df_subnormals, 'output/optimized_lmul_normalxsubnormal.png')