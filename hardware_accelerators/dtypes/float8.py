class Float8:
    MAX_NORMAL = 448.0
    MIN_NORMAL = '0.0001.000'
    MAX_SUBNORMAL = '0.0000.111'
    MIN_SUBNORMAL = '0.0000.001'

    def __init__(self, value, decimal=None):
        # If value is already Float8, copy its attributes directly
        if isinstance(value, Float8):
            self.binary = value.binary
            self.decimal = value.decimal
            self.decimal_approx = value.decimal_approx
            return
            
        if isinstance(value, (float, int)):
            # Handle infinities
            if value > self.MAX_NORMAL:
                value = float('inf')
            if value < -self.MAX_NORMAL:
                value = float('-inf')

            self.decimal = float(value)
            self.binary = self._decimal_to_binary(self.decimal)
            # Remove dots before converting back to decimal
            self.decimal_approx = self._binary_to_decimal(''.join(c for c in self.binary if c in '01'))
        elif isinstance(value, str):
            # Remove whitespace and punctuation
            cleaned = ''.join(c for c in value if c in '01')
            
            # Validate binary string
            if len(cleaned) != 8 or not all(c in '01' for c in cleaned):
                raise ValueError("Binary string must be exactly 8 bits of 0's and 1's")
            
            self.binary = cleaned[0]+'.'+cleaned[1:5]+'.'+cleaned[5:]
            self.decimal_approx = self._binary_to_decimal(cleaned)
            if decimal is not None:
                self.decimal = decimal
            else:
                self.decimal = self.decimal_approx
        else:
            raise TypeError("Float8 must be initialized with float, int, or binary string")

    @classmethod
    def from_binint(cls, binint):
        binary = format(binint, '08b')
        # Create instance and ensure decimal values are set
        instance = cls(binary)
        if instance.decimal is None:
            instance.decimal = instance.decimal_approx
        return instance

    def __mul__(self, other):
        # Convert other to Float8 if it's a float or int
        if isinstance(other, (float, int)):
            other = Float8(other)
        
        # Check if other is Float8
        if not isinstance(other, Float8):
            return NotImplemented
        
        # Ensure both operands have valid decimal values
        if self.decimal is None:
            self.decimal = self.decimal_approx
        if other.decimal is None:
            other.decimal = other.decimal_approx
            
        # Perform multiplication using decimal values
        result = self.decimal * other.decimal
        return Float8(result)
    
    ######
    
    @property
    def bits(self):
        return ''.join(c for c in self.binary if c in '01')
    
    @property
    def binint(self):
        return int(self.bits, 2)

    def _binary_to_decimal(self, binary):
        binary = ''.join(c for c in binary if c in '01')
        # Extract components
        sign = -1 if binary[0] == '1' else 1
        exp = binary[1:5]
        mantissa = binary[5:]
        
        # Handle special cases
        if exp == '1111' and mantissa == '111':
            return sign * float('inf')
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
        
        # Handle inf and -inf
        if num == float('inf'):
            return sign + ".1111.111"
        
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

    def __add__(self, other):
        if not isinstance(other, (Float8, float, int)):
            return NotImplemented
        
        if isinstance(other, (float, int)):
            other = Float8(other)
            
        result = self.decimal + other.decimal
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