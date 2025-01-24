import pyrtl
from pyrtl import WireVector, Register
from ..dtypes.base import BaseFloat
from dataclasses import dataclass
from typing import Type, Self


@dataclass
class PEOutputs:
    """Container for PE outputs to make connections clear"""

    data: Register  # Passes to PE to the right
    weight: Register  # Passes to PE below
    accum: Register  # Passes to PE below


class ProcessingElement:
    def __init__(
        self,
        data_type: Type[BaseFloat],
        accum_type: Type[BaseFloat],
        multiplier_type,
        adder_type,
        pipeline_mult: bool = False,
    ):
        """Initialize processing element hardware

        Args:
            data_type: Float type for data/weight (Float8, BF16 etc)
            accum_type: Float type for accumulation
            multiplier_type: Floating point multiplier implementation
            adder_type: Floating point adder implementation
            pipeline_mult: If True, register multiplication output before passing to accumulator

        TODO: Add control logic for:
        - Activation flow control
        - Accumulator reset/clear
        """
        # Get bit widths from format specs
        data_width = data_type.bitwidth()
        accum_width = accum_type.bitwidth()

        # Input/output registers
        self.data_reg = Register(bitwidth=data_width)
        self.weight_reg = Register(bitwidth=data_width)
        self.accum_in = WireVector(bitwidth=accum_width)
        self.accum_reg = Register(bitwidth=accum_width)

        # Control signals
        self.weight_we = WireVector(bitwidth=1)  # Weight write enable

        # Multiply-accumulate logic
        product = multiplier_type(
            self.data_reg,
            self.weight_reg,
            data_type.exponent_bits(),
            data_type.mantissa_bits(),
        )

        # TODO: Add float type conversion logic to pass different bitwidths to the accumulator

        if pipeline_mult:
            product_reg = Register(bitwidth=accum_width)
            product_reg.next <<= product
            sum_result = adder_type(
                product_reg,
                self.accum_in,
                accum_type.exponent_bits(),
                accum_type.mantissa_bits(),
            )
        else:
            sum_result = adder_type(
                product,
                self.accum_in,
                accum_type.exponent_bits(),
                accum_type.mantissa_bits(),
            )

        self.accum_reg.next <<= sum_result

        # Store registers in output container
        self.outputs = PEOutputs(
            data=self.data_reg, weight=self.weight_reg, accum=self.accum_reg
        )

    def connect_data(self, source: Self | WireVector):
        """Connect data input from source (PE or external input)"""
        if isinstance(source, ProcessingElement):
            self.data_reg.next <<= source.outputs.data
        else:
            self.data_reg.next <<= source

    def connect_weight(self, source: Self | WireVector):
        """Connect weight input from source (PE or external input)"""
        if isinstance(source, ProcessingElement):
            weight_in = source.outputs.weight
        else:
            weight_in = source

        # Conditional weight update based on enable signal
        with pyrtl.conditional_assignment:
            with self.weight_we:
                self.weight_reg.next |= weight_in

    def connect_accum(self, source: Self | WireVector):
        """Connect accumulator input from source (PE or external input)"""
        if isinstance(source, ProcessingElement):
            self.accum_in <<= source.outputs.accum
        else:
            self.accum_in <<= source

    def connect_weight_enable(self, enable: WireVector):
        """Connect weight write enable signal"""
        self.weight_we <<= enable
