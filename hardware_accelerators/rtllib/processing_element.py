from dataclasses import dataclass
from typing import Callable, Self, Type
import pyrtl
from pyrtl import Register, WireVector, conditional_assignment

from ..dtypes.base import BaseFloat
from .utils.common import convert_float_format


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
        weight_type: Type[BaseFloat],
        accum_type: Type[BaseFloat],
        multiplier: Callable[[WireVector, WireVector, Type[BaseFloat]], WireVector],
        adder: Callable[[WireVector, WireVector, Type[BaseFloat]], WireVector],
        pipeline_mult: bool = False,
    ):
        """Initialize processing element hardware

        Args:
            data_type: Float type for data/weight (Float8, BF16 etc)
            accum_type: Float type for accumulation
            multiplier_type: Floating point multiplier implementation
            adder_type: Floating point adder implementation
            pipeline_mult: If True, register multiplication output before passing to accumulator
        """
        self.data_type = data_type
        self.weight_type = weight_type
        self.accum_type = accum_type
        self.pipeline = pipeline_mult

        # Get bit widths from format specs
        data_width = data_type.bitwidth()
        weight_width = weight_type.bitwidth()
        acc_width = accum_type.bitwidth()

        # Input wires
        self.data_in = WireVector(data_width)
        self.weight_in = WireVector(weight_width)
        self.accum_in = WireVector(acc_width)

        # Control signals
        self.weight_en = WireVector(1)  # Weight write enable
        self.data_en = WireVector(1)  # Enable writing to the data input register
        self.adder_en = WireVector(1)  # Enable writing to the accumulator register

        # Registers
        self.data_reg = Register(data_width)
        self.weight_reg = Register(weight_width)
        self.accum_reg = Register(acc_width)

        # Convert inputs to multiplier if necessary
        multiplier_data_input = self.data_reg
        multiplier_weight_input = self.weight_reg
        multiplier_dtype = data_type

        if data_width < weight_width:
            multiplier_data_input = convert_float_format(
                self.data_reg, data_type, weight_type
            )
            multiplier_dtype = weight_type
        elif weight_width < data_width:
            multiplier_weight_input = convert_float_format(
                self.weight_reg, weight_type, data_type
            )

        # Multiply logic
        product = multiplier(
            multiplier_data_input, multiplier_weight_input, multiplier_dtype
        )

        # Optionally build a pipeline register to hold the multiplier result
        if self.pipeline:
            self.mul_en = WireVector(1)
            product_reg = Register(multiplier_dtype.bitwidth())
            product_out = product_reg
            with conditional_assignment:
                with self.mul_en:  # Enable writing to product register
                    product_reg.next |= product
        else:
            product_out = product

        self.adder_input = convert_float_format(
            product_out, multiplier_dtype, accum_type
        )

        # Add the product and previous accumulation value to get partial sum
        sum_result = adder(self.adder_input, self.accum_in, accum_type)

        # Enable writing to data input register
        with conditional_assignment:
            with self.data_en:
                self.data_reg.next |= self.data_in

        # Enable writing to weight input register
        with conditional_assignment:
            with self.weight_en:
                self.weight_reg.next |= self.weight_in

        # Enable writing to accumulator register
        with conditional_assignment:
            with self.adder_en:
                self.accum_reg.next |= sum_result

        # Store registers in output container
        self.outputs = PEOutputs(
            data=self.data_reg, weight=self.weight_reg, accum=self.accum_reg
        )

    def connect_data(self, source: Self | WireVector):
        """Connect data input from source (PE or external input)"""
        if isinstance(source, ProcessingElement):
            self.data_in <<= source.outputs.data
        else:
            self.data_in <<= source

    def connect_weight(self, source: Self | WireVector):
        """Connect weight input from source (PE or external input)"""
        if isinstance(source, ProcessingElement):
            self.weight_in <<= source.outputs.weight
        else:
            self.weight_in <<= source

    def connect_accum(self, source: Self | WireVector):
        """Connect accumulator input from source (PE or external input)"""
        if isinstance(source, ProcessingElement):
            self.accum_in <<= source.outputs.accum
        else:
            self.accum_in <<= source

    def connect_weight_enable(self, enable: WireVector):
        """Connect weight write enable signal"""
        self.weight_en <<= enable

    def connect_data_enable(self, enable: WireVector):
        """Connect PE enable signal. Controls writing to the data input register"""
        self.data_en <<= enable

    def connect_mul_enable(self, enable: WireVector):
        """Connect multiplier enable signal. Controls writing to the product register"""
        if self.pipeline:
            self.mul_en <<= enable
        else:
            print(
                "Pipelining disabled, no product register to enable. Deleting wire.",
            )
            pyrtl.working_block().remove_wirevector(enable)

    def connect_adder_enable(self, enable: WireVector):
        """Connect adder enable signal. Controls writing to the accumulator register"""
        self.adder_en <<= enable

    def connect_control_signals(
        self,
        weight_en: WireVector | None = None,
        data_en: WireVector | None = None,
        mul_en: WireVector | None = None,
        adder_en: WireVector | None = None,
    ):
        """Connect control signals to the processing element

        Args:
            weight_en (WireVector): Weight write enable signal. Controls writing to the weight register
            data_en (WireVector): PE enable signal. Controls writing to the data input register
            mul_en (WireVector): Multiplier enable signal. Controls writing to the product register
            adder_en (WireVector): Adder enable signal. Controls writing to the accumulator register
        """
        if data_en is not None:
            self.connect_data_enable(data_en)
        if weight_en is not None:
            self.connect_weight_enable(weight_en)
        if mul_en is not None:
            self.connect_mul_enable(mul_en)
        if adder_en is not None:
            self.connect_adder_enable(adder_en)

        return self
