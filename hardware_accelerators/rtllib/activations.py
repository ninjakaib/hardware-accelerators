from typing import Sequence, Type
from pyrtl import (
    WireVector,
    Input,
    Output,
    Register,
    Const,
    select,
    conditional_assignment,
    otherwise,
    Simulation,
)
from ..dtypes.base import BaseFloat


class ReluUnit:
    def __init__(self, size: int, dtype: Type[BaseFloat]):
        self.size = size
        self.dtype = dtype

        # Control signals
        self.start = WireVector(1)  # trigger to latch new enable value
        self.enable_in = WireVector(1)  # input enable value to latch
        self.inputs_valid = WireVector(1)  # indicates if inputs are valid
        self.enable_reg = Register(1)  # stateful enable register

        # Input and output data
        self.data = [WireVector(dtype.bitwidth()) for _ in range(size)]
        self.outputs = [self.relu(x) for x in self.data]
        self.outputs_valid = WireVector(1)
        self.outputs_valid <<= self.inputs_valid

    def relu(self, x: WireVector):
        # Use enable_reg instead of enable wire
        pass_condition = self.inputs_valid & (
            ~self.enable_reg | (self.enable_reg & ~x[-1])
        )
        return select(pass_condition, x, Const(0, self.dtype.bitwidth()))

    def connect_inputs(
        self,
        inputs: list[WireVector] | list[Input],
        enable: WireVector | Input,
        valid: WireVector | Input,
        start: WireVector | Input,
    ):
        assert (
            len(inputs) == self.size
        ), f"Activation module input size mismatch. Expected {self.size}, got {len(inputs)}"
        for i in range(self.size):
            self.data[i] <<= inputs[i]
        self.inputs_valid <<= valid
        self.enable_in <<= enable
        self.start <<= start

        # Update enable register logic
        with conditional_assignment:
            with self.start:
                self.enable_reg.next |= self.enable_in
            with otherwise:
                self.enable_reg.next |= self.enable_reg

    def connect_outputs(
        self,
        outputs: list[WireVector] | list[Output],
        valid: WireVector | Output | None = None,
    ):
        assert (
            len(outputs) == self.size
        ), f"Activation module output size mismatch. Expected {self.size}, got {len(outputs)}"
        for i, out in enumerate(outputs):
            out <<= self.outputs[i]
        if valid is not None:
            assert len(valid) == 1, "Valid output must be a single bit wire"
            valid <<= self.outputs_valid

    def inspect_outputs(self, sim: Simulation) -> list[float]:
        """Inspect current outputs of the ReLU unit.

        Args:
            sim: PyRTL simulation object

        Returns:
            List of float output values in the current simulation step
        """
        return [float(self.dtype(binint=sim.inspect(out.name))) for out in self.outputs]


# class ReluUnit:
#     def __init__(self, size: int, dtype: Type[BaseFloat]):
#         self.size = size
#         self.dtype = dtype
#         self.inputs_valid = WireVector(1)  # indicates if inputs are valid
#         self.enable = WireVector(1)  # enable activation function, otherwise passthrough
#         self.data = [WireVector(dtype.bitwidth()) for _ in range(size)]
#         self.outputs = [self.relu(x) for x in self.data]

#     def relu(self, x: WireVector):
#         pass_condition = self.inputs_valid & (~self.enable | (self.enable & ~x[-1]))
#         return select(pass_condition, x, Const(0, self.dtype.bitwidth()))

#     def connect_inputs(
#         self,
#         inputs: list[WireVector],
#         enable: WireVector,
#         valid: WireVector,
#     ):
#         assert (
#             len(inputs) == self.size
#         ), f"Activation module input size mismatch. Expected {self.size}, got {len(inputs)}"
#         for i in range(self.size):
#             self.data[i] <<= inputs[i]
#         self.inputs_valid <<= valid
#         self.enable <<= enable

#     def connect_outputs(self, outputs: list[WireVector]):
#         assert (
#             len(outputs) == self.size
#         ), f"Activation module output size mismatch. Expected {self.size}, got {len(outputs)}"
#         for i in range(self.size):
#             outputs[i] <<= self.outputs[i]


# Example usage
# ins = input_list([f"in_{i}" for i in range(4)], 4)
# en = Input(1, "en")
# valid = Input(1, "valid")
# outs = output_list([f"out_{i}" for i in range(4)], 4)

# act = ReluUnit(4, FP4)

# act.connect_inputs(ins, en, valid)
# for i in range(4):
#     outs[i] <<= act.outputs[i]
