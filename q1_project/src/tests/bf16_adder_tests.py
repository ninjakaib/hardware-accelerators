# import unittest
# import sys 
# import os
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# from IPython.display import display_svg
# from typing import List, Tuple
# import random
# import torch
# import numpy as np
# import pandas as pd
# from itertools import combinations_with_replacement
# from typing import Callable, Any
# import pyrtl
# from pyrtl.rtllib.libutils import twos_comp_repr, rev_twos_comp_repr
# from pyrtl import (
#     WireVector, 
#     Const, 
#     Input,
#     Output, 
#     Register, 
#     Simulation, 
#     SimulationTrace, 
#     reset_working_block
# )
# from src.utils import custom_render_trace, basic_circuit_analysis
# from src.bfloat16 import BF16
# from src.bf16_adder import *
# from src.repr_funcs import *
# import unittest
# import pyrtl
# import torch
# import numpy as np
# from src.bf16_adder import *

# E_BITS  = 8
# M_BITS  = 7
# MSB     = E_BITS + M_BITS

# class TestBF16AdderHelper:
#     """Helper class containing common test setup and utility functions"""
    
#     @staticmethod
#     def setup_simulation():
#         """Reset PyRTL and create a new simulation trace"""
#         # pyrtl.reset_working_block()
#         sim_trace = pyrtl.SimulationTrace()
#         sim = pyrtl.Simulation(tracer=sim_trace)
#         return sim

#     @staticmethod
#     def generate_test_cases(n_cases=5):
#         """Generate test cases using random BF16 values"""
#         inputs = torch.rand(2, n_cases, dtype=torch.bfloat16)
#         outputs = inputs[0] + inputs[1]
        
#         raw_inputs = inputs.view(torch.uint16)
#         raw_outputs = outputs.view(torch.uint16)
        
#         return raw_inputs.tolist(), raw_outputs.tolist()
    
#     @staticmethod
#     def extract_components(bf16_val):
#         """Extract sign, exponent, and mantissa from a BF16 value"""
#         sign = (bf16_val >> 15) & 0x1
#         exp = (bf16_val >> 7) & 0xFF
#         mant = bf16_val & 0x7F
#         return sign, exp, mant

#     @staticmethod
#     def compare_components(val1, val2):
#         """Compare two BF16 values by their components"""
#         s1, e1, m1 = TestBF16AdderHelper.extract_components(val1)
#         s2, e2, m2 = TestBF16AdderHelper.extract_components(val2)
#         return s1 == s2 and e1 == e2 and m1 == m2

# class TestBF16Adder(unittest.TestCase):
#     def setUp(self):
#         """Setup common test resources"""
#         pyrtl.reset_working_block()
#         self.helper = TestBF16AdderHelper()
#         self.test_cases = self.helper.generate_test_cases()
    
#     def test_stage1_extraction(self):
#         """Test stage 1: Component extraction"""
        
#         # Create inputs
#         float_a = pyrtl.Input(16, 'float_a')
#         float_b = pyrtl.Input(16, 'float_b')
        
#         # Run stage 1
#         sign_a, sign_b, exp_a, exp_b, mant_a, mant_b = stage_1(float_a, float_b, E_BITS, M_BITS)
#         sim = self.helper.setup_simulation()
#         # Create simulation inputs
#         test_a, test_b = self.test_cases[0][0], self.test_cases[0][1]
#         sim.step({'float_a': test_a, 'float_b': test_b})
        
#         # Verify components
#         for val, sign, exp, mant in [(test_a, 'sign_a', 'exp_a', 'mantissa_a'),
#                                    (test_b, 'sign_b', 'exp_b', 'mantissa_b')]:
#             s, e, m = self.helper.extract_components(val)
#             self.assertEqual(sim.inspect(sign), s)
#             self.assertEqual(sim.inspect(exp), e)
#             # Add 1 for implicit leading 1
#             self.assertEqual(sim.inspect(mant), (m | 0x80))

#     def test_stage2_comparison(self):
#         """Test stage 2: Exponent comparison and shift calculation"""
        
#         # Create test inputs
#         sign_a = pyrtl.Input(1, 'sign_a')
#         sign_b = pyrtl.Input(1, 'sign_b')
#         exp_a = pyrtl.Input(E_BITS, 'exp_a')
#         exp_b = pyrtl.Input(E_BITS, 'exp_b')
#         mant_a = pyrtl.Input(M_BITS+1, 'mant_a')
#         mant_b = pyrtl.Input(M_BITS+1, 'mant_b')
        
#         # Run stage 2
#         sign_xor, exp_larger, signed_shift, mant_smaller, mant_larger = stage_2(
#             sign_a, sign_b, exp_a, exp_b, mant_a, mant_b, E_BITS, M_BITS
#         )
        
#         sim = self.helper.setup_simulation()
#         # Test cases
#         test_inputs = {
#             'sign_a': [0, 1, 0],
#             'sign_b': [0, 0, 1],
#             'exp_a': [128, 129, 127],
#             'exp_b': [127, 128, 128],
#             'mant_a': [0x80, 0x82, 0x80],
#             'mant_b': [0x80, 0x80, 0x84]
#         }
        
#         for i in range(len(test_inputs['sign_a'])):
#             step_inputs = {k: v[i] for k, v in test_inputs.items()}
#             sim.step(step_inputs)
            
#             # Verify sign XOR
#             expected_sign_xor = step_inputs['sign_a'] ^ step_inputs['sign_b']
#             self.assertEqual(sim.inspect(sign_xor.name), expected_sign_xor)
            
#             # Verify larger exponent selection
#             exp_diff = step_inputs['exp_a'] - step_inputs['exp_b']
#             expected_larger = step_inputs['exp_a'] if exp_diff >= 0 else step_inputs['exp_b']
#             self.assertEqual(sim.inspect(exp_larger.name), expected_larger)
            
#             # Verify shift amount
#             expected_shift = abs(exp_diff)
#             self.assertEqual(sim.inspect(signed_shift.name) & 0xFF, expected_shift)

#     def test_stage3_alignment(self):
#         """Test stage 3: Mantissa alignment and SGR bit generation"""
        
#         # Create test inputs
#         mant_smaller = pyrtl.Input(M_BITS + 1, 'mant_smaller')
#         shift_amount = pyrtl.Input(E_BITS, 'shift_amount')
        
#         # Run stage 3
#         aligned_msb, sticky, guard, round_bit = stage_3(
#             mant_smaller, shift_amount, M_BITS
#         )
#         sim = self.helper.setup_simulation()
        
#         # Test cases
#         test_inputs = {
#             'mant_smaller': [
#                 0b10000000,  # No shift needed
#                 0b10100000,  # Small shift
#                 0b11000000,  # Large shift
#                 0b10010000   # Shift with sticky bits
#             ],
#             'shift_amount': [
#                 0,    # No shift
#                 2,    # Small shift
#                 8,    # Large shift 
#                 4     # Medium shift
#             ]
#         }
        
#         expected_results = [
#             # [aligned_msb, sticky, guard, round]
#             [0b10000000, 0, 0, 0],  # No shift
#             [0b00100000, 0, 0, 0],  # Small shift
#             [0b00000000, 1, 0, 0],  # Large shift (all bits become sticky)
#             [0b00001001, 0, 0, 0]   # Medium shift with some sticky bits
#         ]
        
#         for i in range(len(test_inputs['mant_smaller'])):
#             sim.step({
#                 'mant_smaller': test_inputs['mant_smaller'][i],
#                 'shift_amount': test_inputs['shift_amount'][i]
#             })
            
#             # Verify aligned mantissa and SGR bits
#             self.assertEqual(sim.inspect(aligned_msb.name), expected_results[i][0])
#             self.assertEqual(sim.inspect(sticky.name), expected_results[i][1])
#             self.assertEqual(sim.inspect(guard.name), expected_results[i][2])
#             self.assertEqual(sim.inspect(round.name), expected_results[i][3])

#     def test_stage4_addition(self):
#         """Test stage 4: Mantissa addition and LZD"""
        
#         # Create test inputs
#         aligned_msb = pyrtl.Input(M_BITS + 1, 'aligned_msb')
#         mant_larger = pyrtl.Input(M_BITS + 1, 'mant_larger')
#         sign_xor = pyrtl.Input(1, 'sign_xor')
        
#         # Run stage 4
#         mant_sum, is_neg, lzc = stage_4(
#             aligned_msb, mant_larger, sign_xor, M_BITS
#         )
        
#         sim = self.helper.setup_simulation()
#         # Test cases
#         test_inputs = {
#             'aligned_msb': [
#                 0b10000000,  # Normal addition
#                 0b11000000,  # Addition with carry
#                 0b10000000,  # Subtraction case
#                 0b01000000   # Leading zeros case
#             ],
#             'mant_larger': [
#                 0b10000000,  # Normal addition
#                 0b10000000,  # Addition with carry
#                 0b11000000,  # Larger number
#                 0b10000000   # Normal number
#             ],
#             'sign_xor': [
#                 0,  # Addition
#                 0,  # Addition
#                 1,  # Subtraction
#                 0   # Addition
#             ]
#         }
        
#         expected_results = [
#             # [mant_sum, is_neg, lzc]
#             [0b100000000, 0, 0],  # Normal addition
#             [0b101000000, 0, 0],  # Addition with carry
#             [0b001000000, 0, 2],  # Subtraction with leading zeros
#             [0b011000000, 0, 1]   # Result with leading zero
#         ]
        
#         for i in range(len(test_inputs['aligned_msb'])):
#             sim.step({
#                 'aligned_msb': test_inputs['aligned_msb'][i],
#                 'mant_larger': test_inputs['mant_larger'][i],
#                 'sign_xor': test_inputs['sign_xor'][i]
#             })
            
#             # Verify mantissa sum, sign, and leading zero count
#             self.assertEqual(sim.inspect(mant_sum.name), expected_results[i][0])
#             self.assertEqual(sim.inspect(is_neg.name), expected_results[i][1])
#             self.assertEqual(sim.inspect(lzc.name), expected_results[i][2])

#     def test_stage5_normalization(self):
#         """Test stage 5: Normalization, rounding and final assembly"""
        
#         # Create test inputs
#         mant_sum = pyrtl.Input(M_BITS + 2, 'mant_sum')
#         sticky = pyrtl.Input(1, 'sticky')
#         guard = pyrtl.Input(1, 'guard')
#         round_bit = pyrtl.Input(1, 'round_bit')
#         lzc = pyrtl.Input(4, 'lzc')
#         exp_larger = pyrtl.Input(E_BITS, 'exp_larger')
#         sign_a = pyrtl.Input(1, 'sign_a')
#         sign_b = pyrtl.Input(1, 'sign_b')
#         signed_shift = pyrtl.Input(E_BITS + 1, 'signed_shift')
#         is_neg = pyrtl.Input(1, 'is_neg')
        
#         # Run stage 5
#         final_sign, final_exp, norm_mantissa = stage_5(
#             mant_sum, sticky, guard, round_bit, lzc, exp_larger,
#             sign_a, sign_b, signed_shift, is_neg, E_BITS, M_BITS
#         )
        
#         sim = self.helper.setup_simulation()
#         # Test cases
#         test_inputs = {
#             'mant_sum': [
#                 0b100000000,  # Normal case
#                 0b010000000,  # Needs normalization
#                 0b100000001,  # Needs rounding
#                 0b011111111   # Round with carry
#             ],
#             'sticky': [0, 0, 1, 1],
#             'guard': [0, 0, 1, 1],
#             'round_bit': [0, 0, 0, 1],
#             'lzc': [0, 1, 0, 0],
#             'exp_larger': [128, 128, 128, 128],
#             'sign_a': [0, 0, 0, 0],
#             'sign_b': [0, 0, 0, 0],
#             'signed_shift': [0, 0, 0, 0],
#             'is_neg': [0, 0, 0, 0]
#         }
        
#         expected_results = [
#             # [final_sign, final_exp, norm_mantissa]
#             [0, 128, 0b1000000],  # Normal case
#             [0, 127, 0b1000000],  # Normalized case
#             [0, 128, 0b1000001],  # Rounded case
#             [0, 129, 0b1000000]   # Rounded with carry
#         ]
        
#         for i in range(len(test_inputs['mant_sum'])):
#             step_inputs = {k: v[i] for k, v in test_inputs.items()}
#             sim.step(step_inputs)
            
#             # Verify final components
#             self.assertEqual(sim.inspect(final_sign.name), expected_results[i][0])
#             self.assertEqual(sim.inspect(final_exp.name), expected_results[i][1])
#             self.assertEqual(sim.inspect(norm_mantissa.name), expected_results[i][2])

#     def test_full_pipeline(self):
#         """Test complete pipeline functionality"""
        
#         # Create inputs and outputs
#         float_a = pyrtl.Input(16, 'float_a')
#         float_b = pyrtl.Input(16, 'float_b')
#         w_en = pyrtl.Input(1, 'write_enable')
#         result = pyrtl.Output(16, 'result')
        
#         # Instantiate pipeline
#         adder = PipelinedBF16Adder(float_a, float_b, w_en)
#         result <<= adder._result_out
        
#         sim = self.helper.setup_simulation()
#         # Generate test cases
#         ins, outs = self.helper.generate_test_cases(n_cases=10)
#         pipeline_delay = 5  # Number of pipeline stages
        
#         # Create input dictionary with pipeline delay padding
#         inputs = {
#             'float_a': ins[0] + [0] * pipeline_delay,
#             'float_b': ins[1] + [0] * pipeline_delay,
#             'write_enable': [1] * (len(ins[0]) + pipeline_delay)
#         }
        
#         # Run simulation
#         for i in range(len(inputs['float_a'])):
#             sim.step({
#                 'float_a': inputs['float_a'][i],
#                 'float_b': inputs['float_b'][i],
#                 'write_enable': inputs['write_enable'][i]
#             })
            
#             # Check results after pipeline delay
#             if i >= pipeline_delay:
#                 expected = outs[i - pipeline_delay]
#                 result = sim.inspect(result.name)
#                 # Compare components to handle floating point comparison
#                 self.assertTrue(
#                     self.helper.compare_components(result, expected),
#                     f"Mismatch at step {i}: got {result}, expected {expected}"
#                 )

#     def test_pipeline_write_enable(self):
#         """Test pipeline write enable functionality"""
#         sim = self.helper.setup_simulation()
        
#         # Create inputs and outputs
#         float_a = pyrtl.Input(16, 'float_a')
#         float_b = pyrtl.Input(16, 'float_b')
#         w_en = pyrtl.Input(1, 'write_enable')
#         result = pyrtl.Output(16, 'result')
        
#         # Instantiate pipeline
#         adder = PipelinedBF16Adder(float_a, float_b, w_en)
#         result <<= adder._result_out
        
#         # Generate test cases
#         ins, outs = self.helper.generate_test_cases(n_cases=5)
        
#         # Test with alternating write enable
#         inputs = {
#             'float_a': ins[0],
#             'float_b': ins[1],
#             'write_enable': [1, 0, 1, 0, 1]  # Alternating enable/disable
#         }
        
#         # Run simulation
#         for i in range(len(inputs['float_a'])):
#             sim.step({
#                 'float_a': inputs['float_a'][i],
#                 'float_b': inputs['float_b'][i],
#                 'write_enable': inputs['write_enable'][i]
#             })
            
#             # After pipeline delay, verify output is 0 when write_enable was 0
#             if i >= 4:  # After pipeline delay
#                 if inputs['write_enable'][i-4] == 0:
#                     self.assertEqual(sim.inspect(result.name), 0)

# if __name__ == '__main__':
#     unittest.main()


# def create_bf16_adder_tests(
#     n_values: int = 5,
#     pipeline_stages: int = 0
# ) -> tuple[dict[str, list[int]]]:
    
#     inputs = torch.rand(2, n_values, dtype=torch.bfloat16)
#     outputs = inputs[0] + inputs[1]
    
#     if pipeline_stages > 0:
#         inputs = torch.concat((inputs, torch.zeros(2, pipeline_stages, dtype=torch.bfloat16)), dim=1)
#         outputs = torch.concat((torch.zeros(pipeline_stages, dtype=torch.bfloat16), outputs))

#     raw_inputs = inputs.view(torch.uint16)
#     raw_outputs = outputs.view(torch.uint16)

#     return raw_inputs.tolist(), raw_outputs.tolist()

# def test_bfloat16_adder_combinatorial():
#     reset_working_block()

#     float_a, float_b, gt = Input(16, 'float_a'), Input(16, 'float_b'), Input(16, 'gt')
#     float_out = Output(16, 'float_out')

#     float_out <<= bfloat16_adder_combinatorial(float_a, float_b, E_BITS, M_BITS)

#     sim_trace = SimulationTrace()
#     sim = Simulation(tracer=sim_trace)

#     ins, outs = create_bf16_adder_tests()
#     test_inputs = {
#         'float_a': ins[0],
#         'float_b': ins[1],
#         'gt': outs
#     }

#     sim.step_multiple(test_inputs)

#     trace_list = ['float_a', 'float_b', 'gt', 'float_out']

#     custom_render_trace(sim_trace, trace_list=trace_list, repr_func=repr_bf16_tensor)

# def test_pipelined_adder(n_tests: int = 5, stages: int = 4):
#     reset_working_block()
#     print("Testing pipelined BF16 adder:")

#     # Define inputs
#     a, b, w_en, gt = Input(16, 'float_a'), Input(16, 'float_b'), Input(1, 'write_enable'), Input(16, 'gt')
#     result = Output(16, 'result')

#     # Instantiate the pipeline
#     adder = PipelinedBF16Adder(a,b,w_en)
#     result <<= adder._result_out
    
#     # basic_circuit_analysis(tech_in_nm=32)
#     # Create simulation
#     sim_trace = SimulationTrace()
#     sim = Simulation(tracer=sim_trace)

#     ins, outs = create_bf16_adder_tests(n_values=n_tests, pipeline_stages=stages)
#     inputs = {
#         'float_a': ins[0],
#         'float_b': ins[1],
#         'write_enable': [1] * n_tests + [0] * stages,
#         'gt': outs
#     }
    
#     # Run simulation
#     sim.step_multiple(inputs)
    
#     # Display results
#     input_repr_map = {
#         'float_a': repr_bf16_tensor,
#         'float_b': repr_bf16_tensor,
#         'result': repr_bf16_tensor,
#         'gt': repr_bf16_tensor,
#         'write_enable': bool
#     }
    
#     trace_list = list(input_repr_map.keys())
#     custom_render_trace(sim_trace, trace_list=trace_list, repr_per_name=input_repr_map)

# def test_stage1(e_bits=E_BITS, m_bits=M_BITS):
#     reset_working_block()
#     total_bits = 1 + e_bits + m_bits
#     # Inputs should match the bfloat16 format: 1 sign bit, 8 exponent bits, and 7 mantissa bits
#     input_a = Input(total_bits, name='input_a')
#     input_b = Input(total_bits, name='input_b')

#     # Extract components
#     stage_1(input_a, input_b, e_bits, m_bits)
    
#     # Simulate and test
#     sim_trace = SimulationTrace()
#     sim = Simulation(tracer=sim_trace)
#     sim.step({'input_a': 0b0100000001000000, 'input_b': 0b0100000010000000})  # Example inputs

#     input_repr_map = {
#         'input_a': repr_bf16,
#         'input_b': repr_bf16,
#         'sign_a': repr_sign,
#         'sign_b': repr_sign,
#         'exp_a': repr_exp,
#         'exp_b': repr_exp,
#         'mantissa_a': repr_mantissa,
#         'mantissa_b': repr_mantissa,
#     }
#     trace_list = list(input_repr_map.keys())

#     custom_render_trace(sim_trace, trace_list=trace_list, repr_per_name=input_repr_map)

# def test_stage_2():
#     reset_working_block()

#     sign_a, sign_b  = Input(1, 'sign_a')        , Input(1, 'sign_b')
#     exp_a, exp_b    = Input(E_BITS, 'exp_a')    , Input(E_BITS, 'exp_b')
#     mant_a, mant_b  = Input(M_BITS+1, 'mant_a') , Input(M_BITS+1, 'mant_b')

#     sxr, exp_larger, shift_amount, mant_smaller, mant_larger = stage_2(sign_a, sign_b, exp_a, exp_b, mant_a, mant_b, E_BITS, M_BITS)

#     sxr_out = Output(1, 'sxr_out')
#     exp_larger_out = Output(E_BITS, 'exp_larger_out')
#     shift_amount_out = Output(E_BITS+1, 'shift_amount_out')
#     mant_smaller_out = Output(M_BITS+1, 'mant_smaller_out')
#     mant_larger_out = Output(M_BITS+1, 'mant_larger_out')

#     sxr_out <<= sxr
#     exp_larger_out <<= exp_larger
#     shift_amount_out <<= shift_amount
#     mant_smaller_out <<= mant_smaller
#     mant_larger_out <<= mant_larger
    
#     sim_trace = pyrtl.SimulationTrace()
#     sim = pyrtl.Simulation(tracer=sim_trace)

#     inputs = {
#         'sign_a': [0,0,0,0],
#         'sign_b': [0,0,0,0],
#         'exp_a': [1, 128, 129, 255],
#         'exp_b': [128, 255, 128, 200],
#         'mant_a': [129, 133, 200, 192],
#         'mant_b': [160, 192, 160, 148]
#     }
#     sim.step_multiple(inputs)  # Example inputs

#     # Representation maps for visualization
#     def repr_signed_shift(x):
#         binary = format(x, f'0{E_BITS+1}b')
#         sign = '(+)' if binary[0]=='0' else '(-)'
#         return f"{sign} {binary[1:]} ({int(binary[1:], 2)})"
    
#     input_repr_map = {
#         'exp_a': repr_exp,
#         'exp_b': repr_exp,
#         'mant_a': repr_mantissa,
#         'mant_b': repr_mantissa,
#         'exp_larger': repr_exp,
#         'exp_diff': lambda x: rev_twos_comp_repr(x, E_BITS+1),
#         'signed_shift': repr_signed_shift,
#         'mant_smaller': repr_mantissa,
#         'mant_larger': repr_mantissa,
#     }
#     trace_list = list(input_repr_map.keys())

#     custom_render_trace(sim_trace, trace_list=trace_list, repr_per_name=input_repr_map, repr_func=repr_num)

# def test_align_mantissa():
#     reset_working_block()
    
#     mant_smaller = Input(M_BITS + 1, 'mant_smaller')
#     shift_amount = Input(E_BITS, 'shift_amount')
    
#     aligned_msb, aligned_lsb = align_mantissa(mant_smaller, shift_amount, M_BITS)
#     aligned_msb_out = Output(M_BITS+1, 'aligned_msb_out')
#     aligned_lsb_out = Output(M_BITS+1, 'aligned_lsb_out')
#     aligned_msb_out <<= aligned_msb
#     aligned_lsb_out <<= aligned_lsb

#     sim_trace = SimulationTrace()
#     sim = Simulation(tracer=sim_trace)
    
#     inputs = {
#         'mant_smaller': [0b10000000, 0b10100000, 0b11000000, 0b10000000],
#         'shift_amount': [1, 9, 3, 45]
#     }
    
#     sim.step_multiple(inputs)
#     input_repr_map = {
#         'mant_smaller': repr_mantissa,
#         'shift_amount': repr_num,
#         'clamped_shift': str,
#         'aligned_mantissa': repr_ext_mantissa,
#         'aligned_msb_out': repr_mantissa,
#         'aligned_lsb_out': lambda x: format(x, f'0{M_BITS+1}b'),
#     }
    
#     trace_list = list(input_repr_map.keys())
#     custom_render_trace(sim_trace, trace_list=trace_list, repr_per_name=input_repr_map)

# def test_generate_sgr():
#     reset_working_block()
    
#     aligned_mant_lsb = Input(M_BITS + 1, 'aligned_mant_lsb')
    
#     sticky_bit, guard_bit, round_bit = generate_sgr(aligned_mant_lsb, M_BITS)
    
#     sticky_out = Output(1, 'sticky_out')
#     guard_out = Output(1, 'guard_out')
#     round_out = Output(1, 'round_out')
    
#     sticky_out <<= sticky_bit
#     guard_out <<= guard_bit
#     round_out <<= round_bit
    
#     sim_trace = SimulationTrace()
#     sim = Simulation(tracer=sim_trace)
    
#     inputs = {
#         'aligned_mant_lsb': [0b10000000, 0b01000000, 0b10100000, 0b11000000, 0b00100000],
#     }
    
#     sim.step_multiple(inputs)
    
#     input_repr_map = {
#         'aligned_mant_lsb': lambda x: format(x, f'0{M_BITS+1}b'),
#         'sticky_out': repr_num,
#         'guard_out': repr_num,
#         'round_out': repr_num
#     }
    
#     trace_list = list(input_repr_map.keys())
#     custom_render_trace(sim_trace, trace_list=trace_list, repr_per_name=input_repr_map)

# def test_add_sub():
#     reset_working_block()
    
#     # Create inputs
#     mant_aligned = Input(M_BITS + 1, 'mant_aligned')
#     mant_unchanged = Input(M_BITS + 1, 'mant_unchanged')
#     sign_xor = Input(1, 'sign_xor')
    
#     # Get outputs
#     result, is_neg = add_sub_mantissas(mant_aligned, mant_unchanged, sign_xor, M_BITS)
    
#     # Create output wires
#     result_out = Output(M_BITS + 2, 'result_out')
#     is_neg_out = Output(1, 'is_neg_out')
    
#     result_out <<= result
#     is_neg_out <<= is_neg
    
#     # Simulate
#     sim_trace = SimulationTrace()
#     sim = Simulation(tracer=sim_trace)
    
#     # Test cases:
#     # 1. Simple addition (same signs)
#     # 2. Simple subtraction (different signs)
#     # 3. Result becomes negative
#     # 4. Large numbers that might overflow
#     inputs = {
#         'mant_aligned':   [0b10000000, 0b10000000, 0b11000000, 0b11111111],
#         'mant_unchanged': [0b10000000, 0b11000000, 0b10000000, 0b11111111],
#         'sign_xor':        [0,          1,          1,          0],
#     }
    
#     sim.step_multiple(inputs)
    
#     input_repr_map = {
#         'mant_aligned': repr_mantissa,
#         'mant_unchanged': repr_mantissa,
#         # 'effective_operation': lambda x: 'Subtract' if x else 'Add',
#         'raw_result': lambda x: format(x, f'0{M_BITS + 3}b'),
#         'result_out': repr_mantissa_sum,
#         'is_neg_out': repr_sign
#     }
    
#     trace_list = list(input_repr_map.keys())
#     custom_render_trace(sim_trace, trace_list=trace_list, repr_per_name=input_repr_map)

# def test_leading_zero_count_8bit():
#     """Test the 8-bit leading zero counter implementation"""
#     pyrtl.reset_working_block()
    
#     input_val = pyrtl.Input(8, 'input_val')
#     lzc_out = pyrtl.Output(4, 'lzc_out')
    
#     lzc_out <<= leading_zero_count_8bit(input_val, M_BITS)
    
#     sim_trace = pyrtl.SimulationTrace()
#     sim = pyrtl.Simulation(tracer=sim_trace)
    
#     # Test vectors for 8-bit values
#     test_values = [
#         0b10000000,  # 0 leading zeros
#         0b01000000,  # 1 leading zero
#         0b00100000,  # 2 leading zeros
#         0b00000001,  # 7 leading zeros
#         0b00000000   # 8 leading zeros
#     ]
    
#     sim.step_multiple({'input_val': test_values})
    
#     # Custom representation for better readability
#     input_repr_map = {
#         'input_val': lambda x: format(x, '08b'),
#         'lzc_out': str,
#     }
    
#     trace_list = list(input_repr_map.keys())
#     custom_render_trace(sim_trace, trace_list=trace_list, repr_per_name=input_repr_map)

# def test_leading_zero_detector_module():
#     """Test the 9-bit leading zero detector module"""
#     reset_working_block()

#     mant_sum_in = Input(9, 'mant_sum_in')
#     leading_zero_detector_module(mant_sum_in, M_BITS)
    
#     sim_trace = pyrtl.SimulationTrace()
#     sim = pyrtl.Simulation(tracer=sim_trace)
    
#     # Test vectors for 9-bit values
#     test_values = [
#         0b100000000,  # 0 leading zeros
#         0b010000000,  # 1 leading zero
#         0b110000000,
#         0b001000000,  # 2 leading zeros
#         0b000001101,
#         0b000000001,  # 8 leading zeros
#         0b000000000   # 9 leading zeros
#     ]
#     sim.step_multiple({'mant_sum_in': test_values})

#     input_repr_map = {
#         'mant_sum_in': repr_mantissa_sum,
#         'leading_zero_count': repr_num,
#     }
#     trace_list = list(input_repr_map.keys())
#     custom_render_trace(sim_trace, trace_list=trace_list, repr_per_name=input_repr_map)

# def test_stage_4():
#     reset_working_block()
    
#     # Create inputs
#     mant_aligned = Input(M_BITS + 1, 'mant_aligned')
#     mant_unchanged = Input(M_BITS + 1, 'mant_unchanged')
#     sign_xor = Input(1, 'sign_xor')
    
#     # Get outputs
#     mantissa_sum, is_neg, lzc = stage_4(mant_aligned, mant_unchanged, sign_xor, M_BITS)
    
#     # Create output wires
#     mant_sum_out = Output(M_BITS + 2, 'mant_sum_out')
#     is_neg_out = Output(1, 'is_neg_out')
#     lzc_out = Output(4, 'lzc_out')
    
#     mant_sum_out <<= mantissa_sum
#     is_neg_out <<= is_neg
#     lzc_out <<= lzc
    
#     # Simulate
#     sim_trace = SimulationTrace()
#     sim = Simulation(tracer=sim_trace)
    
#     # Test cases:
#     # 1. Simple addition (same signs)
#     # 2. Simple subtraction (different signs)
#     # 3. Result becomes negative
#     # 4. Large numbers that might overflow
#     inputs = {
#         'mant_aligned':   [0b10000000, 0b10000000, 0b11000000, 0b11111111, 0b01000000],
#         'mant_unchanged': [0b10000000, 0b11000000, 0b10000000, 0b11111111, 0b10000000],
#         'sign_xor':        [0,          1,          1,          0,           0],
#     }
    
#     sim.step_multiple(inputs)
    
#     input_repr_map = {
#         'mant_aligned': repr_mantissa,
#         'mant_unchanged': repr_mantissa,
#         'mant_sum_out': repr_mantissa_sum,
#         'is_neg_out': repr_sign,
#         'lzc_out': repr_num,
#     }
    
#     trace_list = list(input_repr_map.keys())
#     custom_render_trace(sim_trace, trace_list=trace_list, repr_per_name=input_repr_map)

# def test_detect_final_sign():
#     reset_working_block()
    
#     # Create inputs
#     sign_a = Input(1, 'sign_a')
#     sign_b = Input(1, 'sign_b')
#     exp_diff = Input(E_BITS + 1, 'exp_diff')
#     is_neg = Input(1, 'is_neg')
    
#     # Get output
#     final_sign = detect_final_sign(sign_a, sign_b, exp_diff, is_neg, E_BITS)
    
#     # Create output wire
#     sign_out = Output(1, 'sign_out')
#     sign_out <<= final_sign
    
#     # Simulate
#     sim_trace = SimulationTrace()
#     sim = Simulation(tracer=sim_trace)
    
#     # Test cases covering different scenarios:
#     # 1. Same signs (both positive)
#     # 2. Same signs (both negative)
#     # 3. Different signs, exp_a > exp_b
#     # 4. Different signs, exp_b > exp_a
#     # 5. Different signs, equal exponents
#     inputs = {
#         'sign_a':    [0,    1,    0,    1,    0],
#         'sign_b':    [0,    1,    1,    0,    1],
#         'exp_diff':  [0,    0,    2,    -2,   0],
#         'is_neg':    [0,    1,    0,    1,    1]
#     }
    
#     sim.step_multiple(inputs)
    
#     input_repr_map = {
#         'sign_a': repr_sign,
#         'sign_b': repr_sign,
#         'exp_diff': lambda x: str(rev_twos_comp_repr(x, E_BITS + 1)),
#         'is_neg': repr_sign,
#         'sign_out': repr_sign
#     }
    
#     trace_list = list(input_repr_map.keys())
#     custom_render_trace(sim_trace, trace_list=trace_list, repr_per_name=input_repr_map)

# def test_normalize_and_round():
#     reset_working_block()
    
#     # Create inputs
#     abs_mantissa = Input(M_BITS + 2, 'abs_mantissa')
#     sticky = Input(1, 'sticky')
#     guard = Input(1, 'guard')
#     round_bit = Input(1, 'round')
#     lzc = Input(4, 'lzc')
    
#     # Get outputs
#     final_mantissa, extra_inc = normalize_and_round(
#         abs_mantissa, sticky, guard, round_bit, lzc, M_BITS
#     )
    
#     # Create output wires
#     mant_out = Output(M_BITS, 'mant_out')
#     inc_out = Output(1, 'inc_out')
    
#     mant_out <<= final_mantissa
#     inc_out <<= extra_inc
    
#     # Simulate
#     sim_trace = SimulationTrace()
#     sim = Simulation(tracer=sim_trace)
    
#     # Test cases:
#     # 1. No leading zeros, round up
#     # 2. Two leading zeros, round down
#     # 3. One leading zero, tie case
#     # 4. Three leading zeros, round up
#     # 5. No leading zeros, no rounding needed
#     inputs = {
#         'abs_mantissa': [0b010000000,  # Normal case
#                         0b001100000,  # 2 leading zeros
#                         0b001000000,  # 2 leading zeros
#                         0b000100000,  # 3 leading zeros
#                         0b010000000], # No rounding
#         'sticky':       [1, 0, 0, 1, 0],
#         'guard':        [1, 0, 1, 1, 0],
#         'round':        [0, 0, 0, 0, 0],
#         'lzc':          [0, 2, 1, 3, 1]
#     }
    
#     sim.step_multiple(inputs)
    
#     input_repr_map = {
#         'abs_mantissa': repr_mantissa_sum,
#         'sticky': repr_num,
#         'guard': repr_num,
#         'round': repr_num,
#         'lzc': repr_num,
#         'norm_shift': repr_mantissa_sum,
#         'rounded_mantissa': repr_mantissa_sum,
#         'final_mantissa': repr_mantissa_hidden,
#         'inc_out': repr_num
#     }
    
#     trace_list = list(input_repr_map.keys())
#     custom_render_trace(sim_trace, trace_list=trace_list, repr_per_name=input_repr_map)

# def test_adjust_final_exponent():
#     reset_working_block()
    
#     # Create inputs
#     exp_larger = Input(E_BITS, 'exp_larger')
#     lzc = Input(4, 'lzc')
#     round_inc = Input(1, 'round_inc')
    
#     # Get output
#     final_exp = adjust_final_exponent(exp_larger, lzc, round_inc, E_BITS)
    
#     # Create output wire
#     exp_out = Output(E_BITS, 'exp_out')
#     exp_out <<= final_exp
    
#     # Simulate
#     sim_trace = SimulationTrace()
#     sim = Simulation(tracer=sim_trace)
    
#     # Test cases:
#     # 1. Normal case, no round increment
#     # 2. Normal case with round increment
#     # 3. Large LZC value
#     # 4. Near maximum exponent
#     # 5. Near minimum exponent
#     inputs = {
#         'exp_larger': [128,  128,  130,  254,  3],
#         'lzc':        [2,    2,    5,    0,    2],
#         'round_inc':  [0,    1,    0,    1,    0]
#     }
    
#     sim.step_multiple(inputs)
    
#     input_repr_map = {
#         'exp_larger': repr_exp,
#         'lzc': repr_num,
#         'round_inc': repr_num,
#         'lzc_adjusted': repr_exp,
#         'exp_out': repr_exp
#     }
    
#     trace_list = list(input_repr_map.keys())
#     custom_render_trace(sim_trace, trace_list=trace_list, repr_per_name=input_repr_map)

