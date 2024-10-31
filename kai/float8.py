class Float8:
    def __init__(self, value, decimal=None):
        # If value is already Float8, copy its attributes directly
        if isinstance(value, Float8):
            self.__dict__ = value.__dict__.copy()
            return
        
        self.binary = None
        self.decimal = None
        self.decimal_approx = None
        
        # # Check if value is already Float8
        # if isinstance(value, Float8):
        #     self.binary = value.binary
        #     self.decimal = value.decimal
        #     self.decimal_approx = value.decimal_approx

        if isinstance(value, float) or isinstance(value, int):
            self.decimal = float(value)
            self.binary = self._decimal_to_binary(self.decimal)
            self.decimal_approx = self._binary_to_decimal(self.binary)
        elif isinstance(value, str):
            # Remove whitespace and punctuation
            cleaned = ''.join(c for c in value if c in '01')
            
            # Validate binary string
            if len(cleaned) != 8 or not all(c in '01' for c in cleaned):
                raise ValueError("Binary string must be exactly 8 bits of 0's and 1's")
            
            self.binary = cleaned
            self.decimal_approx = self._binary_to_decimal(cleaned)
            if decimal:
                self.decimal = decimal
            else:
                self.decimal = self.decimal_approx
        else:
            raise TypeError("Float8 must be initialized with float, int, or binary string")

    def bits(self):
        return ''.join(c for c in self.binary if c in '01')

    def _binary_to_decimal(self, binary):
        binary = ''.join(c for c in binary if c in '01')
        # Extract components
        sign = -1 if binary[0] == '1' else 1
        exp = binary[1:5]
        mantissa = binary[5:]
        
        # Handle special cases
        if exp == '1111' and mantissa == '111':
            return float('nan')
        if exp == '0000' and mantissa == '000':
            return 0.0
            
        # Convert biased exponent
        biased_exp = int(exp, 2)
        
        if biased_exp == 0:  # Subnormal number
            actual_exp = -6  # emin
            mantissa_value = int(mantissa, 2) / (2 ** 3)  # Divide by 2^mantissa_bits
            return sign * (2 ** actual_exp) * mantissa_value
        else:  # Normal number
            actual_exp = biased_exp - 7  # Subtract bias
            mantissa_value = 1 + int(mantissa, 2) / (2 ** 3)  # 1 + fraction
            return sign * (2 ** actual_exp) * mantissa_value

    def _decimal_to_binary(self, num):
        if num == 0:
            return "0.0000.000"
        
        # Extract sign bit
        sign = "1" if num < 0 else "0"
        num = abs(num)
        
        # Constants for E4M3
        BIAS = 7
        MAX_EXP = 8
        MIN_EXP = -6
        MANTISSA_BITS = 3
        
        # Handle NaN
        if num != num:  # Python's way to check for NaN
            return sign + ".1111.111"
        
        # Find exponent and normalized mantissa
        exp = 0
        temp = num
        while temp >= 2:
            temp /= 2
            exp += 1
        while temp < 1 and exp > MIN_EXP:
            temp *= 2
            exp -= 1
            
        # Handle subnormal numbers
        if exp < MIN_EXP:
            # Shift mantissa right and adjust
            shift = MIN_EXP - exp
            temp /= (2 ** shift)
            exp = MIN_EXP
            
        # Calculate biased exponent
        if temp < 1:  # Subnormal
            biased_exp = "0000"
        else:  # Normal
            biased_exp = format(exp + BIAS, '04b')
            
        # Calculate mantissa bits
        if temp < 1:  # Subnormal
            mantissa_value = int(temp * (2 ** (MANTISSA_BITS)))
        else:  # Normal
            mantissa_value = int((temp - 1) * (2 ** MANTISSA_BITS))
        
        mantissa = format(mantissa_value, f'0{MANTISSA_BITS}b')
        
        # Combine all parts
        result = '.'.join([sign, biased_exp, mantissa])
        
        return result


    def __repr__(self):
        return f"Float8(binary='{self.binary}', decimal={self.decimal}, decimal_approx={self.decimal_approx})"

    def __str__(self):
        return str(self.binary)

    def __add__(self, other):
        if not isinstance(other, (Float8, float, int)):
            return NotImplemented
        
        if isinstance(other, (float, int)):
            other = Float8(other)
            
        result = self.decimal + other.decimal
        return Float8(result)

    def __mul__(self, other):
        if not isinstance(other, (Float8, float, int)):
            return NotImplemented
            
        if isinstance(other, (float, int)):
            other = Float8(other)
            
        result = self.decimal * other.decimal
        return Float8(result)

    def __sub__(self, other):
        if not isinstance(other, (Float8, float, int)):
            return NotImplemented
            
        if isinstance(other, (float, int)):
            other = Float8(other)
            
        result = self.decimal - other.decimal
        return Float8(result)

    def __truediv__(self, other):
        if not isinstance(other, (Float8, float, int)):
            return NotImplemented
            
        if isinstance(other, (float, int)):
            other = Float8(other)
            
        result = self.decimal / other.decimal
        return Float8(result)

    # Comparison operators
    def __eq__(self, other):
        if not isinstance(other, (Float8, float, int)):
            return NotImplemented
        if isinstance(other, (float, int)):
            other = Float8(other)
        return self.decimal == other.decimal

    def __lt__(self, other):
        if not isinstance(other, (Float8, float, int)):
            return NotImplemented
        if isinstance(other, (float, int)):
            other = Float8(other)
        return self.decimal < other.decimal

    def __le__(self, other):
        if not isinstance(other, (Float8, float, int)):
            return NotImplemented
        if isinstance(other, (float, int)):
            other = Float8(other)
        return self.decimal <= other.decimal

    # Reverse operators
    def __radd__(self, other):
        return self.__add__(other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __rsub__(self, other):
        return Float8(other) - self

    def __rtruediv__(self, other):
        return Float8(other) / self


# def insert_dots(binary_string):
#     result = list(binary_string)
#     result.insert(1, '.')
#     result.insert(5, '.')
#     return ''.join(result)


# def display_sign_steps(bits):
#     sign_bit = bits[0]
#     display(Latex(f"Sign Bit: ${sign_bit}$"))
#     display(Latex(f"$(-1)^{sign_bit} = {'-' if sign_bit=='1' else '+'}1$"))

# def display_exponent_steps(bits, format="E4M3"):
#     if format == "E4M3":
#         exp_bits = bits[1:5]
#         bias = 7
#     else: # E5M2
#         exp_bits = bits[1:6] 
#         bias = 15
        
#     exp_val = int(exp_bits, 2)
#     display(Latex(f"Exponent Bits: ${exp_bits}$"))
#     display(Latex(f"$={exp_bits}_2 = {exp_val}_{{10}}$"))
    
#     if exp_val == 0:
#         # Subnormal case
#         display(Latex(f"Subnormal case: $E = 0$ so use $2^{{1-bias}}$"))
#         display(Latex(f"$2^{{1-{bias}}} = 2^{{{1-bias}}}$"))
#     else:
#         # Normal case
#         display(Latex(f"$2^{{E-bias}} = 2^{{{exp_val}-{bias}}} = 2^{{{exp_val-bias}}}$"))

# def display_mantissa_steps(bits, format="E4M3"):
#     if format == "E4M3":
#         mantissa_bits = bits[5:]
#         m = 3
#     else: # E5M2
#         mantissa_bits = bits[6:]
#         m = 2
        
#     exp_val = int(bits[1:5], 2) if format == "E4M3" else int(bits[1:6], 2)
    
#     display(Latex(f"Mantissa Bits: ${mantissa_bits}$"))
    
#     M = int(mantissa_bits, 2)
#     display(Latex(f"$M = {mantissa_bits}_2 = {M}_{{10}}$"))
    
#     if exp_val == 0:
#         # Subnormal case
#         display(Latex(f"Subnormal case: $(0 + 2^{{-{m}}} × {M})$"))
#         mantissa_val = (0 + 2**(-m) * M)
#         display(Latex(f"$= {mantissa_val}$"))
#     else:
#         # Normal case
#         display(Latex(f"$(1 + 2^{{-{m}}} × {M})$"))
#         mantissa_val = (1 + 2**(-m) * M)
#         display(Latex(f"$= {mantissa_val}$"))

# def splice_bits(bits: str, format="E4M3"):
#     if format == "E4M3":
#         return bits[0] + " " + bits[1:5] + " " + bits[5:]
#     else: # E5M2
#         return  bits[0] + " " + bits[1:6] + " " + bits[6:]

# def display_float8_conversion(bits, format="E4M3", verbose=False):
#     if verbose:
#         display(Latex(f"Converting {format} number: {splice_bits(bits, format)}"))
#         display(Latex("Step 1: Sign"))
#         display_sign_steps(bits)
        
#         display(Latex("Step 2: Exponent"))
#         display_exponent_steps(bits, format)
        
#         display(Latex("Step 3: Mantissa"))
#         display_mantissa_steps(bits, format)
    
#     # Calculate final value
#     sign_bit = bits[0]
#     sign = -1 if sign_bit == '1' else 1
    
#     if format == "E4M3":
#         exp_bits = bits[1:5]
#         mantissa_bits = bits[5:]
#         bias = 7
#         m = 3
#     else:
#         exp_bits = bits[1:6]
#         mantissa_bits = bits[6:]
#         bias = 15
#         m = 2
        
#     exp_val = int(exp_bits, 2)
#     M = int(mantissa_bits, 2)
    
#     if exp_val == 0 and M > 0:
#         # Subnormal
#         value = sign * (2**(1-bias)) * (0 + 2**(-m) * M)
#     elif exp_val == 0 and M == 0:
#         # Zero or negative zero
#         value = 0
#     else:
#         # Normal
#         value = sign * (2**(exp_val-bias)) * (1 + 2**(-m) * M)
        
#     display(Latex(f"Final Value: $\\fbox{{{value}}}$"))
#     return value


# display_float8_conversion('00000011')