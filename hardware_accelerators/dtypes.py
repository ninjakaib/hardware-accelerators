from abc import ABC, abstractmethod
from typing import Union, Optional, Tuple
from dataclasses import dataclass
import struct
from enum import Enum

class FloatFormat(Enum):
    """Enum for different floating-point formats"""
    FLOAT8 = "float8"
    BFLOAT16 = "bfloat16"
    # Can be extended with other formats

@dataclass
class FormatSpec:
    """Specification for a floating-point format"""
    total_bits: int
    exponent_bits: int
    mantissa_bits: int
    bias: int
    max_normal: float
    min_normal: float
    max_subnormal: float
    min_subnormal: float

class BaseFloat(ABC):
    """Base class for custom floating-point number representations"""
    
    def __init__(self, 
                 value: Union[int, float, str, 'BaseFloat', None] = None,
                 binary: Optional[str] = None,
                 binint: Optional[int] = None):
        """
        Initialize floating point number from various input types
        
        Args:
            value: Input value (float, int, string, or another BaseFloat instance)
            binary: Binary string representation
            binint: Integer representing the binary value
        """
        self._format_spec = self._get_format_spec()
        self._original_value = None
        self._binary = None
        self._binint = None
        self._decimal_approx = None
        
        # Initialize based on input type
        if value is not None:
            self._init_from_value(value)
        elif binary is not None:
            self._init_from_binary_string(binary)
        elif binint is not None:
            self._init_from_binint(binint)
        else:
            raise ValueError("Must provide one of: value, binary, or binint")

    @abstractmethod
    def _get_format_spec(self) -> FormatSpec:
        """Return format specification for the specific float type"""
        pass

    @property
    def format_spec(self) -> FormatSpec:
        """Get format specification"""
        return self._format_spec

    def _init_from_value(self, value: Union[int, float, str, 'BaseFloat']):
        """Initialize from a value"""
        if isinstance(value, (int, float)):
            self._original_value = float(value)
            self._binary = self._decimal_to_binary(value)
            self._update_all_representations()
        elif isinstance(value, str):
            if value.startswith('0b'):
                self._init_from_binint(int(value, 2))
            else:
                self._init_from_binary_string(value)
        elif isinstance(value, BaseFloat):
            self._original_value = value.original_value
            self._binary = value.binary
            self._update_all_representations()
        else:
            raise TypeError(f"Unsupported type: {type(value)}")

    def _init_from_binary_string(self, binary: str):
        """Initialize from binary string"""
        # Clean binary string
        binary = ''.join(c for c in binary if c in '01')
        if len(binary) != self._format_spec.total_bits:
            raise ValueError(f"Binary string must be {self._format_spec.total_bits} bits")
        self._binary = binary
        self._update_all_representations()

    def _init_from_binint(self, binint: int):
        """Initialize from binary integer"""
        if binint < 0 or binint >= (1 << self._format_spec.total_bits):
            raise ValueError(f"Binary integer must fit in {self._format_spec.total_bits} bits")
        self._binint = binint
        self._binary = format(binint, f'0{self._format_spec.total_bits}b')
        self._update_all_representations()

    def _update_all_representations(self):
        """Update all internal representations for consistency"""
        if self._binary is not None:
            if self._binint is None:
                self._binint = int(self._binary, 2)
            if self._decimal_approx is None:
                self._decimal_approx = self._binary_to_decimal(self._binary)
        if self._original_value is None:
            self._original_value = self._decimal_approx

    @abstractmethod
    def _decimal_to_binary(self, value: float) -> str:
        """Convert decimal to binary string"""
        pass

    @abstractmethod
    def _binary_to_decimal(self, binary: str) -> float:
        """Convert binary string to decimal approximation"""
        pass

    # Properties
    @property
    def original_value(self) -> float:
        """Get original input value"""
        return self._original_value

    @property
    def decimal_approx(self) -> float:
        """Get decimal approximation"""
        return self._decimal_approx

    @property
    def binary(self) -> str:
        """Get binary string representation"""
        return self._binary

    @property
    def binint(self) -> int:
        """Get integer representation of binary"""
        return self._binint

    # Basic arithmetic operations
    def __add__(self, other):
        if isinstance(other, (int, float)):
            return self.__class__(self.decimal_approx + other)
        elif isinstance(other, BaseFloat):
            return self.__class__(self.decimal_approx + other.decimal_approx)
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return self.__class__(self.decimal_approx * other)
        elif isinstance(other, BaseFloat):
            return self.__class__(self.decimal_approx * other.decimal_approx)
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, (int, float)):
            return self.__class__(self.decimal_approx - other)
        elif isinstance(other, BaseFloat):
            return self.__class__(self.decimal_approx - other.decimal_approx)
        return NotImplemented

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return self.__class__(self.decimal_approx / other)
        elif isinstance(other, BaseFloat):
            return self.__class__(self.decimal_approx / other.decimal_approx)
        return NotImplemented

    # Reverse arithmetic operations
    def __radd__(self, other):
        return self.__add__(other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __rsub__(self, other):
        return self.__class__(other - self.decimal_approx)

    def __rtruediv__(self, other):
        return self.__class__(other / self.decimal_approx)

    # Comparison operations
    def __eq__(self, other):
        if isinstance(other, (int, float)):
            return self.decimal_approx == other
        elif isinstance(other, BaseFloat):
            return self.decimal_approx == other.decimal_approx
        return NotImplemented

    def __lt__(self, other):
        if isinstance(other, (int, float)):
            return self.decimal_approx < other
        elif isinstance(other, BaseFloat):
            return self.decimal_approx < other.decimal_approx
        return NotImplemented

    def __le__(self, other):
        return self < other or self == other

    def __gt__(self, other):
        return not (self <= other)

    def __ge__(self, other):
        return not (self < other)

    def __repr__(self):
        return (f"{self.__class__.__name__}(binary='{self.binary}', "
                f"decimal={self.original_value}, "
                f"decimal_approx={self.decimal_approx})")

    def __str__(self):
        return f"{self.decimal_approx}"

    def __float__(self):
        return self.decimal_approx
    

class Float8(BaseFloat):
    """
    8-bit floating point number with E4M3 format
    - 1 sign bit
    - 4 exponent bits (bias 7)
    - 3 mantissa bits
    """
    
    # Define format spec as class constant
    FORMAT_SPEC = FormatSpec(
        total_bits=8,
        exponent_bits=4,
        mantissa_bits=3,
        bias=7,
        # These values are pre-calculated from their binary representations
        max_normal=2**8 * 1.75,      # from 0.1111.110
        min_normal=2**-6,   # from 0.0001.000
        max_subnormal=2**-6 * (7/8),# from 0.0000.111
        min_subnormal=2**-6 * (1/8) # from 0.0000.001
    )
    
    def _get_format_spec(self) -> FormatSpec:
        return self.FORMAT_SPEC

    def _decimal_to_binary(self, num: float) -> str:
        """Convert decimal number to binary string in E4M3 format"""
        if num == 0:
            return "0.0000.000"
        
        # Extract sign bit
        sign = "1" if num < 0 else "0"
        num = abs(num)
        
        # Handle NaN
        if num != num:  # Python's way to check for NaN
            return sign + ".1111.111"
            
        # Find exponent and normalized mantissa
        exp = 0
        temp = num
        
        # Handle normal numbers
        while temp >= 2 and exp < 15:  # Changed from 8 to 15 (max exp for E4M3)
            temp /= 2
            exp += 1
        while temp < 1 and exp > -6:  # Min exp before bias is -6
            temp *= 2
            exp -= 1
            
        # Handle subnormal numbers
        if exp <= -6:
            # Shift mantissa right and adjust
            shift = -6 - exp
            temp /= (2 ** shift)
            exp = -6
            
        # Calculate biased exponent
        if temp < 1:  # Subnormal
            biased_exp = "0000"
        else:  # Normal
            biased_exp = format(exp + self.format_spec.bias, '04b')
            
        # Calculate mantissa bits
        if temp < 1:  # Subnormal
            mantissa_value = int(temp * (2 ** self.format_spec.mantissa_bits))
        else:  # Normal
            mantissa_value = int((temp - 1) * (2 ** self.format_spec.mantissa_bits))
        
        mantissa = format(mantissa_value, f'0{self.format_spec.mantissa_bits}b')
        
        return f"{sign}.{biased_exp}.{mantissa}"
    
    def _binary_to_decimal(self, binary: str) -> float:
        """Convert binary string in E4M3 format to decimal"""
        # Clean the binary string
        binary = ''.join(c for c in binary if c in '01')
        
        # Extract components
        sign = -1 if binary[0] == '1' else 1
        exp = binary[1:5]
        mantissa = binary[5:]
        
        # Handle special cases
        if exp == '1111' and mantissa == '111':
            return float('nan')
            # For all other mantissa values when exp is '1111',
            # calculate normally instead of returning max value
        if exp == '0000' and mantissa == '000':
            return 0.0
            
        # Convert biased exponent
        biased_exp = int(exp, 2)
        
        if biased_exp == 0:  # Subnormal number
            actual_exp = -6  # emin
            mantissa_value = int(mantissa, 2) / (2 ** self.format_spec.mantissa_bits)
            return sign * (2 ** actual_exp) * mantissa_value
        else:  # Normal number
            actual_exp = biased_exp - self.format_spec.bias
            mantissa_value = 1 + int(mantissa, 2) / (2 ** self.format_spec.mantissa_bits)
            return sign * (2 ** actual_exp) * mantissa_value

    @classmethod
    def from_bits(cls, bits: int) -> 'Float8':
        """Create Float8 from 8-bit integer"""
        return cls(binint=bits)

    @classmethod
    def nan(cls) -> 'Float8':
        """Create NaN value"""
        return cls(binary="1.1111.111")

    @classmethod
    def max_value(cls) -> 'Float8':
        """Create maximum representable value"""
        return cls(binary="0.1111.110")

    @classmethod
    def min_value(cls) -> 'Float8':
        """Create minimum representable normal value"""
        return cls(binary="0.0001.000")

    @classmethod
    def min_subnormal(cls) -> 'Float8':
        """Create minimum representable subnormal value"""
        return cls(binary="0.0000.001")
    
    def detailed_breakdown(self) -> dict:
        """
        Provide detailed breakdown of the Float8 number components
        """
        binary = ''.join(c for c in self.binary if c in '01')
        sign_bit = int(binary[0])
        exp_bits = binary[1:5]
        mantissa_bits = binary[5:]
        
        exp_val = int(exp_bits, 2)
        mantissa_val = int(mantissa_bits, 2)
        
        is_normal = exp_val != 0 and exp_val != 15
        is_subnormal = exp_val == 0 and mantissa_val != 0
        is_zero = exp_val == 0 and mantissa_val == 0
        is_nan = exp_val == 15 and mantissa_val == 7  # Only s.1111.111 is NaN
        
        return {
            'binary': self.binary,
            'sign': sign_bit,
            'exponent_bits': exp_bits,
            'exponent_value': exp_val - self.format_spec.bias if exp_val != 0 else 'subnormal',
            'mantissa_bits': mantissa_bits,
            'mantissa_value': mantissa_val,
            'decimal_approx': self.decimal_approx,
            'original_value': self.original_value,
            'is_normal': is_normal,
            'is_subnormal': is_subnormal,
            'is_zero': is_zero,
            'is_nan': is_nan,
            'normalized_value': (1 + mantissa_val / 8) if is_normal else (mantissa_val / 8),
        }

class BF16(BaseFloat):
    """
    16-bit Brain Floating Point (bfloat16)
    - 1 sign bit
    - 8 exponent bits (bias 127)
    - 7 mantissa bits
    """
    
    FORMAT_SPEC = FormatSpec(
        total_bits=16,
        exponent_bits=8,
        mantissa_bits=7,
        bias=127,
        # max_normal: exp=11111110 (254-127=127), mantissa=1111111 (1 + 127/128)
        max_normal=2**127 * (1 + 127/128),
        # min_normal: exp=00000001 (1-127=-126), mantissa=0000000 (1 + 0/128)
        min_normal=2**-126 * (1 + 0/128),
        # max_subnormal: exp=00000000 (-126), mantissa=1111111 (127/128)
        max_subnormal=2**-126 * (127/128),
        # min_subnormal: exp=00000000 (-126), mantissa=0000001 (1/128)
        min_subnormal=2**-126 * (1/128)
    )
    
    def _get_format_spec(self) -> FormatSpec:
        return self.FORMAT_SPEC

    def _float32_to_bf16_parts(self, f32: float) -> Tuple[int, int, int]:
        """Convert float32 to BF16 parts (sign, exponent, mantissa)"""
        # Get binary representation of float32
        bits = struct.unpack('!I', struct.pack('!f', f32))[0]
        
        # Extract parts from float32
        sign = (bits >> 31) & 0x1
        exp = (bits >> 23) & 0xFF
        mantissa = (bits >> 16) & 0x7F  # Keep only top 7 bits of mantissa
        
        return sign, exp, mantissa

    def _bf16_parts_to_float32(self, sign: int, exp: int, mantissa: int) -> float:
        """Convert BF16 parts back to float32"""
        # Construct 32-bit representation
        bits = (sign << 31) | (exp << 23) | (mantissa << 16)
        return struct.unpack('!f', struct.pack('!I', bits))[0]

    def _decimal_to_binary(self, num: float) -> str:
        """Convert decimal number to binary string in BF16 format"""
        if num == 0:
            return "0" * 16
            
        sign, exp, mantissa = self._float32_to_bf16_parts(num)
        
        # Handle special cases
        if exp == 0xFF:
            if mantissa == 0:
                # Infinity
                return f"{sign}{'1' * 8}{'0' * 7}"
            else:
                # NaN
                return f"{sign}{'1' * 8}{'1' * 7}"
                
        # Construct binary string
        return f"{sign:01b}{exp:08b}{mantissa:07b}"

    def _binary_to_decimal(self, binary: str) -> float:
        """Convert binary string in BF16 format to decimal"""
        # Clean binary string
        binary = ''.join(c for c in binary if c in '01')
        
        # Extract components
        sign = int(binary[0], 2)
        exp = int(binary[1:9], 2)
        mantissa = int(binary[9:], 2)
        
        return self._bf16_parts_to_float32(sign, exp, mantissa)

    @classmethod
    def from_float32(cls, f32: float) -> 'BF16':
        """Create BF16 from float32 value"""
        return cls(f32)

    @classmethod
    def inf(cls, sign: bool = False) -> 'BF16':
        """Create infinite BF16 value"""
        return cls(binary=f"{'1' if sign else '0'}{'1' * 8}{'0' * 7}")

    @classmethod
    def nan(cls, quiet: bool = True) -> 'BF16':
        """
        Create NaN BF16 value
        quiet: If True, creates quiet NaN, else signaling NaN
        """
        if quiet:
            return cls(binary=f"0{'1' * 8}1{'1' * 6}")
        return cls(binary=f"0{'1' * 8}0{'1' * 6}")

    @classmethod
    def max_value(cls) -> 'BF16':
        """Create maximum finite value"""
        return cls(binary=f"0{'1' * 7}0{'1' * 7}")

    @classmethod
    def min_normal(cls) -> 'BF16':
        """Create minimum normal value"""
        return cls(binary=f"0{'0' * 7}1{'0' * 7}")

    @classmethod
    def min_subnormal(cls) -> 'BF16':
        """Create minimum subnormal value"""
        return cls(binary=f"0{'0' * 8}{'0' * 6}1")

    def detailed_breakdown(self) -> dict:
        """
        Provide detailed breakdown of the BF16 number components
        """
        binary = self.binary
        sign_bit = int(binary[0])
        exp_bits = binary[1:9]
        mantissa_bits = binary[9:]
        
        exp_val = int(exp_bits, 2)
        mantissa_val = int(mantissa_bits, 2)
        
        return {
            'binary': binary,
            'sign': sign_bit,
            'exponent_bits': exp_bits,
            'exponent_value': exp_val - self._format_spec.bias if exp_val != 0 else 'subnormal',
            'mantissa_bits': mantissa_bits,
            'mantissa_value': mantissa_val,
            'decimal_approx': self.decimal_approx,
            'original_value': self.original_value,
            'is_normal': exp_val != 0 and exp_val != 255,
            'is_subnormal': exp_val == 0 and mantissa_val != 0,
            'is_zero': exp_val == 0 and mantissa_val == 0,
            'is_inf': exp_val == 255 and mantissa_val == 0,
            'is_nan': exp_val == 255 and mantissa_val != 0,
        }

    def hex_representation(self) -> str:
        """Get hexadecimal representation"""
        return f"0x{self.binint:04X}"