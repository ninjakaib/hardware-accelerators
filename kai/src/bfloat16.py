import torch

class BF16:
    def __init__(self, value, binary_literal=False):
        """
        Initialize a BF16 number from various input types
        """
        # Handle different input types
        if isinstance(value, (int, float)) and not binary_literal:
            self.tensor = torch.tensor(value, dtype=torch.bfloat16)
            self.original_value = value
        elif isinstance(value, torch.Tensor):
            self.tensor = value.to(torch.bfloat16)
            self.original_value = float(value)
        elif isinstance(value, BF16):
            self.tensor = value.tensor
            self.original_value = value.original_value
        elif isinstance(value, str):
            # Binary string initialization
            value = ''.join([c for c in value if c in '01'])
            if not all(c in '01' for c in value):
                raise ValueError("Binary string must contain only 0s and 1s")
            
            # Ensure the string is 16 bits long, pad if necessary
            if len(value) > 16:
                raise ValueError("Binary string cannot be longer than 16 bits")
            
            # Zero-pad to 16 bits if shorter
            value = value.zfill(16)
            
            # Convert binary string to integer
            raw_bits = int(value, 2)
            
            # Convert to BF16 tensor
            self.tensor = torch.tensor(raw_bits, dtype=torch.uint16).view(torch.bfloat16)
            self.original_value = float(self.tensor)
        elif isinstance(value, int) and binary_literal:
            # Integer representation initialization
            # Ensure the integer fits in 16 bits
            if value < 0 or value > 0xFFFF:
                raise ValueError("Integer must be between 0 and 65535 (0xFFFF)")
            
            # Convert integer to BF16 tensor
            self.tensor = torch.tensor(value, dtype=torch.uint16).view(torch.bfloat16)
            self.original_value = float(self.tensor)
        else:
            raise TypeError(f"Cannot convert {type(value)} to BF16")
    
    def __repr__(self):
        """String representation of the BF16 number"""
        return f"{self.tensor}"
        return f"bf16({float(self.tensor)})"
    
    def __float__(self):
        """Convert BF16 to float"""
        return float(self.tensor)
    
    def __mul__(self, other):
        """
        Multiply two BF16 values
        Supports multiplication with:
        - Another BF16 instance
        - int
        - float
        - torch.Tensor
        """
        # Handle different input types
        if isinstance(other, (int, float)):
            # Multiply with scalar
            result_tensor = self.tensor * other
            return BF16(result_tensor)
        elif isinstance(other, torch.Tensor):
            # Multiply with tensor
            result_tensor = self.tensor * other.to(torch.bfloat16)
            return BF16(result_tensor)
        elif isinstance(other, type(self)):
            # Multiply with another BF16 instance
            result_tensor = self.tensor * other.tensor
            return BF16(result_tensor)
        # If multiplication is not supported, return NotImplemented
        return NotImplemented
    
    def __add__(self, other):
        """
        Add two BF16 values
        Supports addition with:
        - Another BF16 instance
        - int
        - float
        - torch.Tensor
        """
        if isinstance(other, (int, float)):
            # Add with scalar
            result_tensor = self.tensor + torch.tensor(other, dtype=torch.bfloat16)
            return BF16(result_tensor)
        elif isinstance(other, torch.Tensor):
            # Add with tensor
            result_tensor = self.tensor + other.to(torch.bfloat16)
            return BF16(result_tensor)
        elif isinstance(other, type(self)):
            # Add with another BF16 instance
            result_tensor = self.tensor + other.tensor
            return BF16(result_tensor)
        # If addition is not supported, return NotImplemented
        return NotImplemented

    def __radd__(self, other):
        """
        Reverse addition (when BF16 is the right operand)
        """
        return self.__add__(other)

    def __sub__(self, other):
        """
        Subtract two BF16 values
        Supports subtraction with:
        - Another BF16 instance
        - int
        - float
        - torch.Tensor
        """
        if isinstance(other, (int, float)):
            # Subtract scalar
            result_tensor = self.tensor - torch.tensor(other, dtype=torch.bfloat16)
            return BF16(result_tensor)
        elif isinstance(other, torch.Tensor):
            # Subtract tensor
            result_tensor = self.tensor - other.to(torch.bfloat16)
            return BF16(result_tensor)
        elif isinstance(other, type(self)):
            # Subtract another BF16 instance
            result_tensor = self.tensor - other.tensor
            return BF16(result_tensor)
        # If subtraction is not supported, return NotImplemented
        return NotImplemented

    def __rsub__(self, other):
        """
        Reverse subtraction (when BF16 is the right operand)
        """
        if isinstance(other, (int, float)):
            # other - self
            result_tensor = torch.tensor(other, dtype=torch.bfloat16) - self.tensor
            return BF16(result_tensor)
        elif isinstance(other, torch.Tensor):
            # tensor - self
            result_tensor = other.to(torch.bfloat16) - self.tensor
            return BF16(result_tensor)
        # If reverse subtraction is not supported, return NotImplemented
        return NotImplemented


    def __sub__(self, other):
        # Handle different input types
        if isinstance(other, (int, float)):
            # Multiply with scalar
            result_tensor = self.tensor - other
            return BF16(result_tensor)
        elif isinstance(other, torch.Tensor):
            # Multiply with tensor
            result_tensor = self.tensor - other.to(torch.bfloat16)
            return BF16(result_tensor)
        elif isinstance(other, type(self)):
            # Multiply with another BF16 instance
            result_tensor = self.tensor - other.tensor
            return BF16(result_tensor)
        # If multiplication is not supported, return NotImplemented
        return NotImplemented
    
    def bit_representation(self):
        """
        Get the full 16-bit representation
        """
        # Convert tensor to uint16 to get raw bits
        raw_bits = self.tensor.view(torch.uint16).item()
        return raw_bits
    
    def detailed_breakdown(self):
        """
        Provide a detailed breakdown of the BF16 representation
        """
        raw_bits = self.bit_representation()
        
        # Bit-level decomposition
        sign_bit = (raw_bits >> 15) & 1
        exponent = (raw_bits >> 7) & 0xFF
        mantissa = raw_bits & 0x7F
        
        return {
            'value': self.tensor,
            'original_value': self.original_value,
            'raw_bits_hex': f'0x{raw_bits:04X}',
            'raw_bits_binary': f'{raw_bits:016b}',
            'sign_bit': sign_bit,
            'exponent_raw': exponent,
            'exponent_biased': exponent - 127,
            'mantissa': mantissa
        }
    
    def to_float32(self):
        """
        Convert to a full precision float32
        """
        return float(self.tensor)
    
    def to_float64(self):
        """
        Convert to a full precision float64
        """
        return float(self.tensor.to(torch.float64))
    
    def hex_representation(self):
        """
        Get hexadecimal representation
        """
        return f'0x{self.bit_representation():04X}'
    
    def binary_representation(self):
        """
        Get binary representation
        """
        return f'{self.bit_representation():016b}'
    
    # Static method for creating special values
    @classmethod
    def inf(cls, sign=False):
        """Create infinite BF16"""
        return cls(float('inf') if not sign else float('-inf'))
    
    @classmethod
    def nan(cls):
        """Create NaN BF16"""
        return cls(float('nan'))
    
    @classmethod
    def from_binary(cls, binary_int):
        """Create BF16 from binary integer"""
        return cls(binary_int, binary_literal=True)

# Demonstration function
def demo_bf16_initialization():
    print("BF16 Initialization Demonstrations:")
    
    # Initialization methods
    demos = [
        # Float/standard initialization
        BF16(3.14),
        BF16(100),
        
        # Binary string initialization
        BF16('0100000001000100'),  # Specific bit pattern
        
        # Integer representation initialization
        BF16(0b0100000001000100, binary_literal=True),  # Binary literal
        BF16(0b0100000001000100),
        BF16(0x4444, True),  # Hex representation
    ]
    
    for num in demos:
        print("\n" + "="*40)
        print(f"Number: {num}")
        
        # Get detailed breakdown
        breakdown = num.detailed_breakdown()
        for key, value in breakdown.items():
            print(f"{key}: {value}")

# Run the demo
if __name__ == "__main__":
    demo_bf16_initialization()