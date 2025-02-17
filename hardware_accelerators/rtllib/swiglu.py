from pyrtl import WireVector
from typing import Type
from .utils.activation_utils import swish
from .lmul import LmulPipelined
from ..dtypes.base import BaseFloat


class SwigluBlock:
    """
    SwiGLU(x, W, V, b, c, β) = Swish_β(xW + b) ⊗ (xV + c)

    Parameters
    ----------
    linear_a: WireVector
        Result of first linear transformation (xW + b).
    linear_b: WireVector
        Result of second linear transformation (xV + c).
    dtype: Type[BaseFloat]
        Floating-point data type.
    """

    def __init__(
        self, linear_a: WireVector, linear_b: WireVector, dtype: Type[BaseFloat]
    ):
        self.linear_a = linear_a
        self.linear_b = linear_b
        self.dtype = dtype

    def compute_swiglu(self) -> WireVector:

        swish_a = swish(self.linear_a, self.dtype)
        result = LmulPipelined(swish_a, self.linear_b, self.dtype)

        return result.output_reg
