from abc import ABC, abstractmethod
from typing import Union, Optional
from dataclasses import dataclass
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

    def __init__(
        self,
        value: Union[int, float, str, "BaseFloat", None] = None,
        binary: Optional[str] = None,
        binint: Optional[int] = None,
    ):
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

    def _init_from_value(self, value: Union[int, float, str, "BaseFloat"]):
        """Initialize from a value"""
        if isinstance(value, (int, float)):
            self._original_value = float(value)
            self._binary = self._decimal_to_binary(value)
            self._update_all_representations()
        elif isinstance(value, str):
            if value.startswith("0b"):
                self._init_from_binint(int(value, 2))
            else:
                self._init_from_binary_string(value)
        elif isinstance(value, BaseFloat):
            self._original_value = value.original_value
            self._binary = value.binary
            self._update_all_representations()
        elif hasattr(value, "__float__"):  # Handle numpy types and other float-like objects
            self._original_value = float(value)
            self._binary = self._decimal_to_binary(float(value))
            self._update_all_representations()
        else:
            raise TypeError(f"Unsupported type: {type(value)}")

    def _init_from_binary_string(self, binary: str):
        """Initialize from binary string"""
        # Clean binary string and format it
        clean_binary = "".join(c for c in binary if c in "01")
        if len(clean_binary) != self._format_spec.total_bits:
            raise ValueError(
                f"Binary string must be {self._format_spec.total_bits} bits"
            )
        self._binary = self._format_binary_string(clean_binary)
        self._update_all_representations()

    def _init_from_binint(self, binint: int):
        """Initialize from binary integer"""
        if binint < 0 or binint >= (1 << self._format_spec.total_bits):
            raise ValueError(
                f"Binary integer must fit in {self._format_spec.total_bits} bits"
            )
        self._binint = binint
        self._binary = format(binint, f"0{self._format_spec.total_bits}b")
        self._update_all_representations()

    def _update_all_representations(self):
        """Update all internal representations for consistency"""
        if self._binary is not None:
            if self._binint is None:
                # Clean the binary string of dots before converting to int
                clean_binary = "".join(c for c in self._binary if c in "01")
                self._binint = int(clean_binary, 2)
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

    def _format_binary_string(self, binary: str) -> str:
        """Format binary string with dots for readability"""
        # Clean the input string first
        clean_binary = "".join(c for c in binary if c in "01")
        if len(clean_binary) != self._format_spec.total_bits:
            raise ValueError(
                f"Binary string must be {self._format_spec.total_bits} bits"
            )

        if self._format_spec.total_bits == 8:  # Float8
            return f"{clean_binary[0]}.{clean_binary[1:5]}.{clean_binary[5:]}"
        elif self._format_spec.total_bits == 16:  # BF16
            return clean_binary  # BF16 doesn't use dot formatting
        else:
            return clean_binary

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
            if other == 0:
                return self.__class__(float("nan"))
            return self.__class__(self.decimal_approx / other)
        elif isinstance(other, BaseFloat):
            if other.decimal_approx == 0:
                return self.__class__(float("nan"))
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
        return (
            f"{self.__class__.__name__}(binary='{self.binary}', "
            f"decimal={self.original_value}, "
            f"decimal_approx={self.decimal_approx})"
        )

    def __str__(self):
        return f"{self.decimal_approx}"

    def __float__(self):
        return self.decimal_approx
