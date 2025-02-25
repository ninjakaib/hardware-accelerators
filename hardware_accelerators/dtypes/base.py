from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Union


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
    """Base class for custom floating-point number representations.
    Classes inheriting from BaseFloat must implement the `format_spec` class method like so:
    ```python
    @classmethod
    def format_spec(cls) -> FormatSpec:
       return FormatSpec(...)
    ```
    """

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

    @classmethod
    @abstractmethod
    def format_spec(cls) -> FormatSpec:
        """Return format specification for the specific float type"""
        pass

    @classmethod
    def bitwidth(cls) -> int:
        return cls.format_spec().total_bits

    @classmethod
    def exponent_bits(cls) -> int:
        return cls.format_spec().exponent_bits

    @classmethod
    def mantissa_bits(cls) -> int:
        return cls.format_spec().mantissa_bits

    @classmethod
    def bias(cls) -> int:
        return cls.format_spec().bias

    @classmethod
    def max_normal(cls) -> float:
        return cls.format_spec().max_normal

    @classmethod
    def min_normal(cls) -> float:
        return cls.format_spec().min_normal

    @classmethod
    def max_subnormal(cls) -> float:
        return cls.format_spec().max_subnormal

    @classmethod
    def min_subnormal(cls) -> float:
        return cls.format_spec().min_subnormal

    def _init_from_value(self, value: Union[int, float, str, "BaseFloat"]):
        """Initialize from a value"""
        if isinstance(value, str):
            if value.startswith("0b"):
                self._init_from_binint(int(value, 2))
            else:
                self._init_from_binary_string(value)
        elif isinstance(value, BaseFloat):
            self._original_value = value.original_value
            self._binary = value.binary
            self._update_all_representations()
        elif hasattr(
            value, "__float__"
        ):  # Handle numpy types and other float-like objects
            self._original_value = float(value)
            self._binary = self._decimal_to_binary(float(value))
            self._update_all_representations()
        else:
            try:
                self._original_value = float(value)
                self._binary = self._decimal_to_binary(value)
                self._update_all_representations()
            except:
                raise TypeError(f"Unsupported type: {type(value)}")

    def _init_from_binary_string(self, binary: str):
        """Initialize from binary string"""
        # Clean binary string and format it
        clean_binary = "".join(c for c in binary if c in "01")
        if len(clean_binary) != self.bitwidth():
            raise ValueError(f"Binary string must be {self.bitwidth()} bits")
        self._binary = self._format_binary_string(clean_binary)
        self._update_all_representations()

    def _init_from_binint(self, binint: int):
        """Initialize from binary integer"""
        if binint < 0 or binint >= (1 << self.bitwidth()):
            raise ValueError(f"Binary integer must fit in {self.bitwidth()} bits")
        self._binint = binint
        self._binary = format(binint, f"0{self.bitwidth()}b")
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

    def _format_binary_string(self, binary=None) -> str:
        """Format binary string with dots for readability"""
        if binary is None:
            binary = self.binary
        clean_binary = "".join(c for c in binary if c in "01")
        if len(clean_binary) != self.bitwidth():
            raise ValueError(f"Binary string must be {self.bitwidth()} bits")

        if self.bitwidth() == 8:  # Float8
            return f"{clean_binary[0]}.{clean_binary[1:5]}.{clean_binary[5:]}"
        elif self.bitwidth() == 32:  # Float32
            return f"{clean_binary[0]}.{clean_binary[1:9]}.{clean_binary[9:]}"
        elif self.bitwidth() == 16:
            if self.__class__.__name__ == "Float16":  # Float16
                return f"{clean_binary[0]}.{clean_binary[1:6]}.{clean_binary[6:]}"
            else:  # BF16
                return clean_binary
        else:
            return clean_binary

    # Properties
    @property
    def original_value(self) -> float:
        """Get original input value"""
        return self._original_value  # type: ignore

    @property
    def decimal_approx(self) -> float:
        """Get decimal approximation"""
        return self._decimal_approx  # type: ignore

    @property
    def binary(self) -> str:
        """Get binary string representation"""
        return self._binary  # type: ignore

    @property
    def binint(self) -> int:
        """Get integer representation of binary"""
        return self._binint  # type: ignore

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

    def __neg__(self):
        return self.__class__(binint=self.binint ^ (1 << self.bitwidth() - 1))
