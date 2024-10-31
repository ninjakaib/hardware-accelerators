def display_sign_steps(bits):
    sign_bit = bits[0]
    display(Latex(f"Sign Bit: ${sign_bit}$"))
    display(Latex(f"$(-1)^{sign_bit} = {'-' if sign_bit=='1' else '+'}1$"))

def display_exponent_steps(bits, format="E4M3"):
    if format == "E4M3":
        exp_bits = bits[1:5]
        bias = 7
    else: # E5M2
        exp_bits = bits[1:6] 
        bias = 15
        
    exp_val = int(exp_bits, 2)
    display(Latex(f"Exponent Bits: ${exp_bits}$"))
    display(Latex(f"$={exp_bits}_2 = {exp_val}_{{10}}$"))
    
    if exp_val == 0:
        # Subnormal case
        display(Latex(f"Subnormal case: $E = 0$ so use $2^{{1-bias}}$"))
        display(Latex(f"$2^{{1-{bias}}} = 2^{{{1-bias}}}$"))
    else:
        # Normal case
        display(Latex(f"$2^{{E-bias}} = 2^{{{exp_val}-{bias}}} = 2^{{{exp_val-bias}}}$"))

def display_mantissa_steps(bits, format="E4M3"):
    if format == "E4M3":
        mantissa_bits = bits[5:]
        m = 3
    else: # E5M2
        mantissa_bits = bits[6:]
        m = 2
        
    exp_val = int(bits[1:5], 2) if format == "E4M3" else int(bits[1:6], 2)
    
    display(Latex(f"Mantissa Bits: ${mantissa_bits}$"))
    
    M = int(mantissa_bits, 2)
    display(Latex(f"$M = {mantissa_bits}_2 = {M}_{{10}}$"))
    
    if exp_val == 0:
        # Subnormal case
        display(Latex(f"Subnormal case: $(0 + 2^{{-{m}}} × {M})$"))
        mantissa_val = (0 + 2**(-m) * M)
        display(Latex(f"$= {mantissa_val}$"))
    else:
        # Normal case
        display(Latex(f"$(1 + 2^{{-{m}}} × {M})$"))
        mantissa_val = (1 + 2**(-m) * M)
        display(Latex(f"$= {mantissa_val}$"))

def splice_bits(bits: str, format="E4M3"):
    if format == "E4M3":
        return bits[0] + " " + bits[1:5] + " " + bits[5:]
    else: # E5M2
        return  bits[0] + " " + bits[1:6] + " " + bits[6:]

def display_float8_conversion(bits, format="E4M3", verbose=False):
    if verbose:
        display(Latex(f"Converting {format} number: {splice_bits(bits, format)}"))
        display(Latex("Step 1: Sign"))
        display_sign_steps(bits)
        
        display(Latex("Step 2: Exponent"))
        display_exponent_steps(bits, format)
        
        display(Latex("Step 3: Mantissa"))
        display_mantissa_steps(bits, format)
    
    # Calculate final value
    sign_bit = bits[0]
    sign = -1 if sign_bit == '1' else 1
    
    if format == "E4M3":
        exp_bits = bits[1:5]
        mantissa_bits = bits[5:]
        bias = 7
        m = 3
    else:
        exp_bits = bits[1:6]
        mantissa_bits = bits[6:]
        bias = 15
        m = 2
        
    exp_val = int(exp_bits, 2)
    M = int(mantissa_bits, 2)
    
    if exp_val == 0 and M > 0:
        # Subnormal
        value = sign * (2**(1-bias)) * (0 + 2**(-m) * M)
    elif exp_val == 0 and M == 0:
        # Zero or negative zero
        value = 0
    else:
        # Normal
        value = sign * (2**(exp_val-bias)) * (1 + 2**(-m) * M)
        
    display(Latex(f"Final Value: $\\fbox{{{value}}}$"))
    return value


display_float8_conversion('00000011')