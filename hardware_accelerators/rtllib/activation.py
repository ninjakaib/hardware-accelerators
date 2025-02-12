from typing import Type
from pyrtl import WireVector, Const, select

from ..dtypes.base import BaseFloat


class ReluUnit:
    def __init__(self, size: int, dtype: Type[BaseFloat]):
        self.size = size
        self.dtype = dtype
        self.inputs_valid = WireVector(1)  # indicates if inputs are valid
        self.enable = WireVector(1)  # enable activation function, otherwise passthrough
        self.data = [WireVector(dtype.bitwidth()) for _ in range(size)]
        self.outputs = [self.relu(x) for x in self.data]

    def relu(self, x: WireVector):
        pass_condition = self.inputs_valid & (~self.enable | (self.enable & ~x[-1]))
        return select(pass_condition, x, Const(0, self.dtype.bitwidth()))

    def connect_inputs(
        self,
        inputs: list[WireVector],
        enable: WireVector,
        valid: WireVector,
    ):
        assert (
            len(inputs) == self.size
        ), f"Activation module input size mismatch. Expected {self.size}, got {len(inputs)}"
        for i in range(self.size):
            self.data[i] <<= inputs[i]
        self.inputs_valid <<= valid
        self.enable <<= enable


# Example usage
# ins = input_list([f"in_{i}" for i in range(4)], 4)
# en = Input(1, "en")
# valid = Input(1, "valid")
# outs = output_list([f"out_{i}" for i in range(4)], 4)

# act = ReluUnit(4, FP4)

# act.connect_inputs(ins, en, valid)
# for i in range(4):
#     outs[i] <<= act.outputs[i]
