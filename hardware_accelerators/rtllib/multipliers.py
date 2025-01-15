from IPython.display import display_svg
from typing import List, Tuple
import random
import torch
import numpy as np
import pandas as pd
from itertools import combinations_with_replacement
from typing import Callable, Any
import pyrtl
from pyrtl.rtllib.libutils import twos_comp_repr, rev_twos_comp_repr
from pyrtl.rtllib import adders
from pyrtl.rtllib import multipliers
from pyrtl import (
    WireVector, 
    Const, 
    Input,
    Output, 
    Register, 
    Simulation, 
    SimulationTrace, 
    reset_working_block
)

from .utils.pipeline import SimplePipeline
from .utils.multiplier_utils import *

# E_BITS  = 8
# M_BITS  = 7



class PipelinedBF16Multiplier(SimplePipeline):
    def __init__(self):
        self._float_a = pyrtl.Input(E_BITS + M_BITS + 1, 'float_a')
        self._float_b = pyrtl.Input(E_BITS + M_BITS + 1, 'float_b')
        self._result = pyrtl.Output(E_BITS + M_BITS + 1, 'result')
        super(PipelinedBF16Multiplier, self).__init__()

    def stage_1(self):
        (self.sign_a, 
         self.sign_b, 
         self.exp_a, 
         self.exp_b, 
         self.mantissa_a, 
         self.mantissa_b) = stage_1(self._float_a, self._float_b, E_BITS, M_BITS)

    def stage_2(self):
        (self.sign_out, 
         self.exp_sum, 
         self.mant_product) = stage_2(self.exp_a, self.exp_b, self.sign_a, self.sign_b, self.mantissa_a, self.mantissa_b)

    def stage_3(self):
        self.sign_out = self.sign_out
        self.mant_product = self.mant_product
        (self.leading_zeros, 
         self.unbiased_exp) = stage_3(self.exp_sum, self.mant_product)

    def stage_4(self):
        (final_exponent, 
         final_mantissa) = stage_4(self.unbiased_exp, self.leading_zeros, self.mant_product)
        self._result <<= pyrtl.concat(self.sign_out, final_exponent, final_mantissa)

def ieee_mult_visualization(output_file):
    reset_working_block()
    float_a, float_b = Input(16, 'float_a'), Input(16, 'float_b')

    sign_a, sign_b, exp_a, exp_b, mantissa_a, mantissa_b = stage_1(float_a, float_b, E_BITS, M_BITS)
    sign_out, exp_sum, mant_product = stage_2(exp_a, exp_b, sign_a, sign_b, mantissa_a, mantissa_b)
    leading_zeros, unbiased_exp = stage_3(exp_sum, mant_product)
    final_exponent, final_mantissa = stage_4(unbiased_exp, leading_zeros, mant_product)
    
    #create traces for exponent and mantissa
    final_exp = Output(8, 'final_exponent')
    final_exp <<= final_exponent

    # final result
    result = Output(16, 'result')
    result <<= pyrtl.concat(sign_out, final_exponent, final_mantissa)

    with open(output_file, 'w') as f:
        pyrtl.visualization.output_to_svg(f)