from dataclasses import dataclass
import numpy as np
from typing import TYPE_CHECKING, Sequence, Type
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


@dataclass
class ReluState:
    start: int
    enable_in: int
    enable_reg: int
    inputs_valid: int
    inputs: np.ndarray
    registers: np.ndarray
    outputs_valid: int
    outputs: np.ndarray

    def __repr__(self) -> str:
        """Pretty print the ReLU state"""
        status = "enabled" if self.enable_reg else "disabled"
        valid_str = "(valid)" if self.outputs_valid else "(invalid)"

        return (
            f"ReLU {status} {valid_str}\n"
            f"  Control: start={self.start}, enable_in={self.enable_in}, "
            f"enable_reg={self.enable_reg}, inputs_valid={self.inputs_valid}\n"
            f"  Inputs: {np.array2string(self.inputs, precision=4, suppress_small=True)}\n"
            f"  Registers: {np.array2string(self.registers, precision=4, suppress_small=True)}\n"
            f"  Outputs: {np.array2string(self.outputs, precision=4, suppress_small=True)}"
        )


class ReluUnit:
    def __init__(self, size: int, dtype: Type[BaseFloat]):
        self.size = size
        self.dtype = dtype

        # Control signals
        self.start = WireVector(1)  # trigger to latch new enable value
        self.enable_in = WireVector(1)  # input enable value to latch
        self.enable_reg = Register(1)  # stateful enable register
        self.inputs_valid = WireVector(1)  # indicates if inputs are valid
        self.valid_reg = Register(1)  # stateful valid register
        self.valid_reg.next <<= self.inputs_valid

        # Input and output data
        self.data_in = [WireVector(dtype.bitwidth()) for _ in range(size)]
        self.data_regs = [Register(dtype.bitwidth()) for _ in range(size)]
        for data, reg in zip(self.data_in, self.data_regs):
            reg.next <<= data

        self.outputs = [self.relu(x) for x in self.data_regs]
        self.outputs_valid = WireVector(1)
        self.outputs_valid <<= self.valid_reg

    def relu(self, x: WireVector):
        # Use enable_reg instead of enable wire
        pass_condition = self.valid_reg & (
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
            self.data_in[i] <<= inputs[i]
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

    def inspect_state(self, sim: Simulation) -> ReluState:
        """Inspect current state of the ReLU unit."""
        return ReluState(
            start=sim.inspect(self.start.name),
            enable_in=sim.inspect(self.enable_in.name),
            enable_reg=sim.inspect(self.enable_reg.name),
            inputs_valid=sim.inspect(self.inputs_valid.name),
            inputs=np.array(
                [
                    float(self.dtype(binint=sim.inspect(inp.name)))
                    for inp in self.data_in
                ]
            ),
            registers=np.array(
                [
                    float(self.dtype(binint=sim.inspect(reg.name)))
                    for reg in self.data_regs
                ]
            ),
            outputs_valid=sim.inspect(self.outputs_valid.name),
            outputs=np.array(
                [
                    float(self.dtype(binint=sim.inspect(out.name)))
                    for out in self.outputs
                ]
            ),
        )


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
