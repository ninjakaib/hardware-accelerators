from pyrtl import WireVector, Const
from typing import Type
from ..adders import FloatAdderPipelined
from ..lmul import LmulPipelined
from ...dtypes.base import BaseFloat
from ...dtypes.float8 import Float8


def sigmoid(input: WireVector, dtype: Type[BaseFloat]) -> WireVector:
    """
    Simple/Naive Sigmoid: 1/(1+2^{-x})

    Where
        - β = 1
        - e is replaced with 2

    Parameters
    ----------
    input: WireVector
        x in the equation
    """
    one = Const(1, dtype.format_spec.bitwidth())
    two = Const(2, dtype.format_spec.bitwidth())

    two_x = WireVector()  # TODO: find a way to calculate 2^{-x}

    w_en = Const(1, dtype.format_spec.bitwidth())
    denominator = FloatAdderPipelined(one, two_x, w_en, dtype)

    inverse = WireVector()  # TODO: find a way to calculate 1 / denominator

    return inverse


def swish(input: WireVector, dtype: Type[BaseFloat]) -> WireVector:
    """
    Swish_β(xW+b) = (xW+b) * sigmoid(β(xW+b))

    Where
        - β = 1
    """
    sigmoid_out = sigmoid(input, dtype)
    return LmulPipelined(input, sigmoid_out, dtype).output_reg


def relu(input: WireVector, dtype: Type[BaseFloat]) -> WireVector:
    """
    ReLU Activation: ReLU(x) = max(0, x)
    """
    sign_bit = input[0]
    zero = Const(0, bitwidth=dtype.format_spec().total_bits)

    relu_out = WireVector(bitwidth=dtype.format_spec().total_bits)

    # my logic is that if sign_bit is negative then return 0 else input
    relu_out <<= (~sign_bit & input) | (sign_bit & zero)
    return relu_out
