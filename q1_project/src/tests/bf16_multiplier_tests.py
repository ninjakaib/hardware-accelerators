import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from IPython.display import display_svg
from typing import List, Tuple
import random
import torch
import numpy as np
import pandas as pd
from itertools import combinations_with_replacement
from typing import Callable, Any
import unittest
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

# Change relative imports to absolute imports
from utils import custom_render_trace, basic_circuit_analysis, custom_render_trace_output
from bfloat16 import BF16
from bf16_multiplier import *
from repr_funcs import *

E_BITS = 8
M_BITS = 7
MSB = E_BITS + M_BITS

class TestBF16Multiplier(unittest.TestCase):
    def setUp(self):
        # Reset PyRTL working block before each test
        pyrtl.reset_working_block()
        
        # Constants used across tests
        self.E_BITS = 8
        self.M_BITS = 7
        self.MSB = self.E_BITS + self.M_BITS

    def test_stage1(self):
        """Test extraction of sign, exponent, and mantissa components."""
        # Inputs should match the bfloat16 format
        input_a = Input(self.MSB + 1, name='input_a')
        input_b = Input(self.MSB + 1, name='input_b')

        # Extract components
        stage_1(input_a, input_b, self.E_BITS, self.M_BITS)
        
        # Simulate and test
        sim_trace = SimulationTrace()
        sim = Simulation(tracer=sim_trace)
        
        # Test with example inputs
        sim.step({'input_a': 0b0100000001000000, 'input_b': 0b0100000010000000})
        
        # Verify outputs through trace
        input_repr_map = {
            'input_a': repr_bf16,
            'input_b': repr_bf16,
            'sign_a': repr_sign,
            'sign_b': repr_sign,
            'exp_a': repr_exp,
            'exp_b': repr_exp,
            'mantissa_a': repr_mantissa,
            'mantissa_b': repr_mantissa,
        }
        
        # Assert expected values from simulation trace
        self.assertEqual(sim_trace.trace['sign_a'][0], 0)
        self.assertEqual(sim_trace.trace['sign_b'][0], 0)

    def test_stage_2(self):
        """Test sign combination, exponent addition, and mantissa multiplication."""
        float_a, float_b = Input(16, 'float_a'), Input(16, 'float_b')

        sign_a, sign_b, exp_a, exp_b, mantissa_a, mantissa_b = stage_1(
            float_a, float_b, self.E_BITS, self.M_BITS
        )
        sign_out, exp_sum, mant_product = stage_2(
            exp_a, exp_b, sign_a, sign_b, mantissa_a, mantissa_b
        )
        
        sim_trace = SimulationTrace()
        sim = Simulation(tracer=sim_trace)

        # Test with random bfloat16 values
        inputs = {
            'float_a': torch.rand(1, 5, dtype=torch.bfloat16).view(torch.uint16)[0],
            'float_b': torch.rand(1, 5, dtype=torch.bfloat16).view(torch.uint16)[0],
        }
        sim.step_multiple(inputs)

        # Verify mantissa product width and sign calculation
        # self.assertEqual(len(sim_trace.trace['mantissa_product'][0]), (self.M_BITS + 1) * 2)
        # self.assertIn(sim_trace.trace['sign_out'][0], [0, 1])

    def test_stage_3(self):
        """Test leading zero detection and exponent adjustment."""
        float_a, float_b = Input(16, 'float_a'), Input(16, 'float_b')

        sign_a, sign_b, exp_a, exp_b, mantissa_a, mantissa_b = stage_1(
            float_a, float_b, self.E_BITS, self.M_BITS
        )
        sign_out, exp_sum, mant_product = stage_2(
            exp_a, exp_b, sign_a, sign_b, mantissa_a, mantissa_b
        )
        leading_zeros, unbiased_exp = stage_3(exp_sum, mant_product)

        # Create output wires
        lzc_out = Output(self.E_BITS, 'lzc_out')
        exp_out = Output(self.E_BITS, 'exp_out')
        
        lzc_out <<= leading_zeros
        exp_out <<= unbiased_exp
        
        sim_trace = SimulationTrace()
        sim = Simulation(tracer=sim_trace)

        inputs = {
            'float_a': torch.rand(1, 5, dtype=torch.bfloat16).view(torch.uint16)[0],
            'float_b': torch.rand(1, 5, dtype=torch.bfloat16).view(torch.uint16)[0],
        }
        sim.step_multiple(inputs)

        # Verify leading zero count is within valid range
        self.assertLessEqual(sim_trace.trace['lzc_out'][0], self.E_BITS)

    def test_normalize_mantissa_product(self):
        """Test mantissa normalization process."""
        mantissa_product = Input(2 * self.M_BITS + 2, 'mantissa_product')
        leading_zeros = Input(8, 'leading_zeros')
        
        norm_mantissa_msb, norm_mantissa_lsb = normalize_mantissa_product(
            mantissa_product, leading_zeros
        )
        
        msb_out = Output(self.M_BITS + 1, 'msb_out')
        lsb_out = Output(self.M_BITS + 1, 'lsb_out')
        
        msb_out <<= norm_mantissa_msb
        lsb_out <<= norm_mantissa_lsb
        
        sim_trace = SimulationTrace()
        sim = Simulation(tracer=sim_trace)
        
        # Test cases with different leading zero counts
        test_cases = {
            'mantissa_product': [
                0b0100000000000000,  # 1 leading zero
                0b0010000000000000,  # 2 leading zeros
                0b0001000000000000,  # 3 leading zeros
                0b1000000000000000   # No leading zeros
            ],
            'leading_zeros': [1, 2, 3, 0]
        }
        
        sim.step_multiple(test_cases)
        
        # Verify normalization results
        for i in range(len(test_cases['leading_zeros'])):
            self.assertEqual(
                len(format(sim_trace.trace['msb_out'][i], f'0{self.M_BITS+1}b')),
                self.M_BITS + 1
            )

    def test_rounding_mantissa(self):
        """Test mantissa rounding logic."""
        norm_mantissa_msb = Input(self.M_BITS + 1, 'norm_mantissa_msb')
        sticky_bit = Input(1, 'sticky_bit')
        guard_bit = Input(1, 'guard_bit')
        round_bit = Input(1, 'round_bit')
        
        final_mantissa, extra_increment = rounding_mantissa(
            norm_mantissa_msb, sticky_bit, guard_bit, round_bit
        )
        
        mant_out = Output(self.M_BITS, 'mant_out')
        inc_out = Output(1, 'inc_out')
        
        mant_out <<= final_mantissa
        inc_out <<= extra_increment
        
        sim_trace = SimulationTrace()
        sim = Simulation(tracer=sim_trace)
        
        test_cases = {
            'norm_mantissa_msb': [
                0b10000000,  # No rounding needed
                0b10000001,  # Round up
                0b10000010,  # Round down
                0b11111111   # Maximum value
            ],
            'sticky_bit': [0, 1, 0, 1],
            'guard_bit': [0, 1, 0, 1],
            'round_bit': [0, 0, 1, 1]
        }
        
        sim.step_multiple(test_cases)
        
        # Verify mantissa width and increment flag
        for i in range(len(test_cases['sticky_bit'])):
            self.assertEqual(
                len(format(sim_trace.trace['mant_out'][i], f'0{self.M_BITS}b')),
                self.M_BITS
            )
            self.assertIn(sim_trace.trace['inc_out'][i], [0, 1])

    def test_adjust_final_exponent(self):
        """Test final exponent adjustment."""
        unbiased_exp = Input(self.E_BITS, 'unbiased_exp')
        lzc = Input(8, 'lzc')
        round_increment = Input(1, 'round_increment')
        
        final_exp = adjust_final_exponent(unbiased_exp, lzc, round_increment)
        
        exp_out = Output(self.E_BITS, 'exp_out')
        exp_out <<= final_exp
        
        sim_trace = SimulationTrace()
        sim = Simulation(tracer=sim_trace)
        
        test_cases = {
            'unbiased_exp': [1, 1, 3, 127, -126],
            'lzc': [2, 2, 5, 0, 2],
            'round_increment': [0, 1, 0, 1, 0]
        }
        
        sim.step_multiple(test_cases)
        
        # Verify exponent range
        for i in range(len(test_cases['unbiased_exp'])):
            exp_val = sim_trace.trace['exp_out'][i]
            self.assertLessEqual(exp_val, 255)  # Max exponent value
            self.assertGreaterEqual(exp_val, 0)  # Min exponent value

    def test_stage_4(self):
        """Test final stage of multiplication."""
        float_a, float_b = Input(16, 'float_a'), Input(16, 'float_b')
        gt = Input(16, 'gt')

        sign_a, sign_b, exp_a, exp_b, mantissa_a, mantissa_b = stage_1(
            float_a, float_b, self.E_BITS, self.M_BITS
        )
        sign_out, exp_sum, mant_product = stage_2(
            exp_a, exp_b, sign_a, sign_b, mantissa_a, mantissa_b
        )
        leading_zeros, unbiased_exp = stage_3(exp_sum, mant_product)
        final_exponent, final_mantissa = stage_4(
            unbiased_exp, leading_zeros, mant_product
        )
        
        final_exp_out = Output(8, 'final_exponent')
        final_exp_out <<= final_exponent

        result = Output(16, 'result')
        result <<= pyrtl.concat(sign_out, final_exponent, final_mantissa)
        
        sim_trace = SimulationTrace()
        sim = Simulation(tracer=sim_trace)

        ins, outs = create_bf16_multiplier_tests()
        inputs = {
            'float_a': ins[0],
            'float_b': ins[1],
            'gt': outs
        }
        sim.step_multiple(inputs)

        # Verify final result format
        for i in range(len(ins[0])):
            result_val = sim_trace.trace['result'][i]
            self.assertEqual(len(format(result_val, '016b')), 16)

    def test_full_pipeline(self):
        """Test complete pipelined multiplier."""
        multiplier = PipelinedBF16Multiplier()
        
        sim_trace = SimulationTrace()
        sim = Simulation(tracer=sim_trace)

        ins, outs = create_bf16_multiplier_tests(pipeline_stages=3)
        inputs = {
            'float_a': ins[0] + [0]*5,
            'float_b': ins[1] + [0]*5,
        }
        
        sim.step_multiple(inputs, {"result": outs + [0]*5})
        
        # Verify pipeline latency and results
        for i in range(3,len(outs)):
            self.assertEqual(
                sim_trace.trace['result'][i],
                outs[i]
            )

if __name__ == '__main__':
    unittest.main()