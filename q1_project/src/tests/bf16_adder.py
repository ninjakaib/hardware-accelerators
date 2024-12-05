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
from ..utils import custom_render_trace, basic_circuit_analysis
from ..bfloat16 import BF16
from ..bf16_adder import *
from ..repr_funcs import *

E_BITS  = 8
M_BITS  = 7
MSB     = E_BITS + M_BITS

def create_bf16_adder_tests(
    n_values: int = 5,
    pipeline_stages: int = 0
) -> tuple[dict[str, list[int]]]:
    
    inputs = torch.rand(2, n_values, dtype=torch.bfloat16)
    outputs = inputs[0] + inputs[1]
    
    if pipeline_stages > 0:
        inputs = torch.concat((inputs, torch.zeros(2, pipeline_stages, dtype=torch.bfloat16)), dim=1)
        outputs = torch.concat((torch.zeros(pipeline_stages, dtype=torch.bfloat16), outputs))

    raw_inputs = inputs.view(torch.uint16)
    raw_outputs = outputs.view(torch.uint16)

    return raw_inputs.tolist(), raw_outputs.tolist()

def test_bfloat16_adder_combinatorial():
    reset_working_block()

    float_a, float_b, gt = Input(16, 'float_a'), Input(16, 'float_b'), Input(16, 'gt')
    float_out = Output(16, 'float_out')

    float_out <<= bfloat16_adder_combinatorial(float_a, float_b, E_BITS, M_BITS)

    sim_trace = SimulationTrace()
    sim = Simulation(tracer=sim_trace)

    ins, outs = create_bf16_adder_tests()
    test_inputs = {
        'float_a': ins[0],
        'float_b': ins[1],
        'gt': outs
    }

    sim.step_multiple(test_inputs)

    trace_list = ['float_a', 'float_b', 'gt', 'float_out']

    custom_render_trace(sim_trace, trace_list=trace_list, repr_func=repr_bf16_tensor)

def test_stage1(e_bits=E_BITS, m_bits=M_BITS):
    reset_working_block()
    total_bits = 1 + e_bits + m_bits
    # Inputs should match the bfloat16 format: 1 sign bit, 8 exponent bits, and 7 mantissa bits
    input_a = Input(total_bits, name='input_a')
    input_b = Input(total_bits, name='input_b')

    # Extract components
    stage_1(input_a, input_b, e_bits, m_bits)
    
    # Simulate and test
    sim_trace = SimulationTrace()
    sim = Simulation(tracer=sim_trace)
    sim.step({'input_a': 0b0100000001000000, 'input_b': 0b0100000010000000})  # Example inputs

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
    trace_list = list(input_repr_map.keys())

    custom_render_trace(sim_trace, trace_list=trace_list, repr_per_name=input_repr_map)

def test_stage_2():
    reset_working_block()

    sign_a, sign_b  = Input(1, 'sign_a')        , Input(1, 'sign_b')
    exp_a, exp_b    = Input(E_BITS, 'exp_a')    , Input(E_BITS, 'exp_b')
    mant_a, mant_b  = Input(M_BITS+1, 'mant_a') , Input(M_BITS+1, 'mant_b')

    sxr, exp_larger, shift_amount, mant_smaller, mant_larger = stage_2(sign_a, sign_b, exp_a, exp_b, mant_a, mant_b, E_BITS, M_BITS)

    sxr_out = Output(1, 'sxr_out')
    exp_larger_out = Output(E_BITS, 'exp_larger_out')
    shift_amount_out = Output(E_BITS+1, 'shift_amount_out')
    mant_smaller_out = Output(M_BITS+1, 'mant_smaller_out')
    mant_larger_out = Output(M_BITS+1, 'mant_larger_out')

    sxr_out <<= sxr
    exp_larger_out <<= exp_larger
    shift_amount_out <<= shift_amount
    mant_smaller_out <<= mant_smaller
    mant_larger_out <<= mant_larger
    
    sim_trace = pyrtl.SimulationTrace()
    sim = pyrtl.Simulation(tracer=sim_trace)

    inputs = {
        'sign_a': [0,0,0,0],
        'sign_b': [0,0,0,0],
        'exp_a': [1, 128, 129, 255],
        'exp_b': [128, 255, 128, 200],
        'mant_a': [129, 133, 200, 192],
        'mant_b': [160, 192, 160, 148]
    }
    sim.step_multiple(inputs)  # Example inputs

    # Representation maps for visualization
    def repr_signed_shift(x):
        binary = format(x, f'0{E_BITS+1}b')
        sign = '(+)' if binary[0]=='0' else '(-)'
        return f"{sign} {binary[1:]} ({int(binary[1:], 2)})"
    
    input_repr_map = {
        'exp_a': repr_exp,
        'exp_b': repr_exp,
        'mant_a': repr_mantissa,
        'mant_b': repr_mantissa,
        'exp_larger': repr_exp,
        'exp_diff': lambda x: rev_twos_comp_repr(x, E_BITS+1),
        'signed_shift': repr_signed_shift,
        'mant_smaller': repr_mantissa,
        'mant_larger': repr_mantissa,
    }
    trace_list = list(input_repr_map.keys())

    custom_render_trace(sim_trace, trace_list=trace_list, repr_per_name=input_repr_map, repr_func=repr_num)

def test_align_mantissa():
    reset_working_block()
    
    mant_smaller = Input(M_BITS + 1, 'mant_smaller')
    shift_amount = Input(E_BITS, 'shift_amount')
    
    aligned_msb, aligned_lsb = align_mantissa(mant_smaller, shift_amount, M_BITS)
    aligned_msb_out = Output(M_BITS+1, 'aligned_msb_out')
    aligned_lsb_out = Output(M_BITS+1, 'aligned_lsb_out')
    aligned_msb_out <<= aligned_msb
    aligned_lsb_out <<= aligned_lsb

    sim_trace = SimulationTrace()
    sim = Simulation(tracer=sim_trace)
    
    inputs = {
        'mant_smaller': [0b10000000, 0b10100000, 0b11000000, 0b10000000],
        'shift_amount': [1, 9, 3, 45]
    }
    
    sim.step_multiple(inputs)
    input_repr_map = {
        'mant_smaller': repr_mantissa,
        'shift_amount': repr_num,
        'clamped_shift': str,
        'aligned_mantissa': repr_ext_mantissa,
        'aligned_msb_out': repr_mantissa,
        'aligned_lsb_out': lambda x: format(x, f'0{M_BITS+1}b'),
    }
    
    trace_list = list(input_repr_map.keys())
    custom_render_trace(sim_trace, trace_list=trace_list, repr_per_name=input_repr_map)

def test_generate_sgr():
    reset_working_block()
    
    aligned_mant_lsb = Input(M_BITS + 1, 'aligned_mant_lsb')
    
    sticky_bit, guard_bit, round_bit = generate_sgr(aligned_mant_lsb, M_BITS)
    
    sticky_out = Output(1, 'sticky_out')
    guard_out = Output(1, 'guard_out')
    round_out = Output(1, 'round_out')
    
    sticky_out <<= sticky_bit
    guard_out <<= guard_bit
    round_out <<= round_bit
    
    sim_trace = SimulationTrace()
    sim = Simulation(tracer=sim_trace)
    
    inputs = {
        'aligned_mant_lsb': [0b10000000, 0b01000000, 0b10100000, 0b11000000, 0b00100000],
    }
    
    sim.step_multiple(inputs)
    
    input_repr_map = {
        'aligned_mant_lsb': lambda x: format(x, f'0{M_BITS+1}b'),
        'sticky_out': repr_num,
        'guard_out': repr_num,
        'round_out': repr_num
    }
    
    trace_list = list(input_repr_map.keys())
    custom_render_trace(sim_trace, trace_list=trace_list, repr_per_name=input_repr_map)

def test_add_sub():
    reset_working_block()
    
    # Create inputs
    mant_aligned = Input(M_BITS + 1, 'mant_aligned')
    mant_unchanged = Input(M_BITS + 1, 'mant_unchanged')
    sign_xor = Input(1, 'sign_xor')
    
    # Get outputs
    result, is_neg = add_sub_mantissas(mant_aligned, mant_unchanged, sign_xor, M_BITS)
    
    # Create output wires
    result_out = Output(M_BITS + 2, 'result_out')
    is_neg_out = Output(1, 'is_neg_out')
    
    result_out <<= result
    is_neg_out <<= is_neg
    
    # Simulate
    sim_trace = SimulationTrace()
    sim = Simulation(tracer=sim_trace)
    
    # Test cases:
    # 1. Simple addition (same signs)
    # 2. Simple subtraction (different signs)
    # 3. Result becomes negative
    # 4. Large numbers that might overflow
    inputs = {
        'mant_aligned':   [0b10000000, 0b10000000, 0b11000000, 0b11111111],
        'mant_unchanged': [0b10000000, 0b11000000, 0b10000000, 0b11111111],
        'sign_xor':        [0,          1,          1,          0],
    }
    
    sim.step_multiple(inputs)
    
    input_repr_map = {
        'mant_aligned': repr_mantissa,
        'mant_unchanged': repr_mantissa,
        # 'effective_operation': lambda x: 'Subtract' if x else 'Add',
        'raw_result': lambda x: format(x, f'0{M_BITS + 3}b'),
        'result_out': repr_mantissa_sum,
        'is_neg_out': repr_sign
    }
    
    trace_list = list(input_repr_map.keys())
    custom_render_trace(sim_trace, trace_list=trace_list, repr_per_name=input_repr_map)

def test_leading_zero_count_8bit():
    """Test the 8-bit leading zero counter implementation"""
    pyrtl.reset_working_block()
    
    input_val = pyrtl.Input(8, 'input_val')
    lzc_out = pyrtl.Output(4, 'lzc_out')
    
    lzc_out <<= leading_zero_count_8bit(input_val, M_BITS)
    
    sim_trace = pyrtl.SimulationTrace()
    sim = pyrtl.Simulation(tracer=sim_trace)
    
    # Test vectors for 8-bit values
    test_values = [
        0b10000000,  # 0 leading zeros
        0b01000000,  # 1 leading zero
        0b00100000,  # 2 leading zeros
        0b00000001,  # 7 leading zeros
        0b00000000   # 8 leading zeros
    ]
    
    sim.step_multiple({'input_val': test_values})
    
    # Custom representation for better readability
    input_repr_map = {
        'input_val': lambda x: format(x, '08b'),
        'lzc_out': str,
    }
    
    trace_list = list(input_repr_map.keys())
    custom_render_trace(sim_trace, trace_list=trace_list, repr_per_name=input_repr_map)

def test_leading_zero_detector_module():
    """Test the 9-bit leading zero detector module"""
    reset_working_block()

    mant_sum_in = Input(9, 'mant_sum_in')
    leading_zero_detector_module(mant_sum_in, M_BITS)
    
    sim_trace = pyrtl.SimulationTrace()
    sim = pyrtl.Simulation(tracer=sim_trace)
    
    # Test vectors for 9-bit values
    test_values = [
        0b100000000,  # 0 leading zeros
        0b010000000,  # 1 leading zero
        0b110000000,
        0b001000000,  # 2 leading zeros
        0b000001101,
        0b000000001,  # 8 leading zeros
        0b000000000   # 9 leading zeros
    ]
    sim.step_multiple({'mant_sum_in': test_values})

    input_repr_map = {
        'mant_sum_in': repr_mantissa_sum,
        'leading_zero_count': repr_num,
    }
    trace_list = list(input_repr_map.keys())
    custom_render_trace(sim_trace, trace_list=trace_list, repr_per_name=input_repr_map)

def test_stage_4():
    reset_working_block()
    
    # Create inputs
    mant_aligned = Input(M_BITS + 1, 'mant_aligned')
    mant_unchanged = Input(M_BITS + 1, 'mant_unchanged')
    sign_xor = Input(1, 'sign_xor')
    
    # Get outputs
    mantissa_sum, is_neg, lzc = stage_4(mant_aligned, mant_unchanged, sign_xor, M_BITS)
    
    # Create output wires
    mant_sum_out = Output(M_BITS + 2, 'mant_sum_out')
    is_neg_out = Output(1, 'is_neg_out')
    lzc_out = Output(4, 'lzc_out')
    
    mant_sum_out <<= mantissa_sum
    is_neg_out <<= is_neg
    lzc_out <<= lzc
    
    # Simulate
    sim_trace = SimulationTrace()
    sim = Simulation(tracer=sim_trace)
    
    # Test cases:
    # 1. Simple addition (same signs)
    # 2. Simple subtraction (different signs)
    # 3. Result becomes negative
    # 4. Large numbers that might overflow
    inputs = {
        'mant_aligned':   [0b10000000, 0b10000000, 0b11000000, 0b11111111, 0b01000000],
        'mant_unchanged': [0b10000000, 0b11000000, 0b10000000, 0b11111111, 0b10000000],
        'sign_xor':        [0,          1,          1,          0,           0],
    }
    
    sim.step_multiple(inputs)
    
    input_repr_map = {
        'mant_aligned': repr_mantissa,
        'mant_unchanged': repr_mantissa,
        'mant_sum_out': repr_mantissa_sum,
        'is_neg_out': repr_sign,
        'lzc_out': repr_num,
    }
    
    trace_list = list(input_repr_map.keys())
    custom_render_trace(sim_trace, trace_list=trace_list, repr_per_name=input_repr_map)

def test_detect_final_sign():
    reset_working_block()
    
    # Create inputs
    sign_a = Input(1, 'sign_a')
    sign_b = Input(1, 'sign_b')
    exp_diff = Input(E_BITS + 1, 'exp_diff')
    is_neg = Input(1, 'is_neg')
    
    # Get output
    final_sign = detect_final_sign(sign_a, sign_b, exp_diff, is_neg, E_BITS)
    
    # Create output wire
    sign_out = Output(1, 'sign_out')
    sign_out <<= final_sign
    
    # Simulate
    sim_trace = SimulationTrace()
    sim = Simulation(tracer=sim_trace)
    
    # Test cases covering different scenarios:
    # 1. Same signs (both positive)
    # 2. Same signs (both negative)
    # 3. Different signs, exp_a > exp_b
    # 4. Different signs, exp_b > exp_a
    # 5. Different signs, equal exponents
    inputs = {
        'sign_a':    [0,    1,    0,    1,    0],
        'sign_b':    [0,    1,    1,    0,    1],
        'exp_diff':  [0,    0,    2,    -2,   0],
        'is_neg':    [0,    1,    0,    1,    1]
    }
    
    sim.step_multiple(inputs)
    
    input_repr_map = {
        'sign_a': repr_sign,
        'sign_b': repr_sign,
        'exp_diff': lambda x: str(rev_twos_comp_repr(x, E_BITS + 1)),
        'is_neg': repr_sign,
        'sign_out': repr_sign
    }
    
    trace_list = list(input_repr_map.keys())
    custom_render_trace(sim_trace, trace_list=trace_list, repr_per_name=input_repr_map)

def test_normalize_and_round():
    reset_working_block()
    
    # Create inputs
    abs_mantissa = Input(M_BITS + 2, 'abs_mantissa')
    sticky = Input(1, 'sticky')
    guard = Input(1, 'guard')
    round_bit = Input(1, 'round')
    lzc = Input(4, 'lzc')
    
    # Get outputs
    final_mantissa, extra_inc = normalize_and_round(
        abs_mantissa, sticky, guard, round_bit, lzc, M_BITS
    )
    
    # Create output wires
    mant_out = Output(M_BITS, 'mant_out')
    inc_out = Output(1, 'inc_out')
    
    mant_out <<= final_mantissa
    inc_out <<= extra_inc
    
    # Simulate
    sim_trace = SimulationTrace()
    sim = Simulation(tracer=sim_trace)
    
    # Test cases:
    # 1. No leading zeros, round up
    # 2. Two leading zeros, round down
    # 3. One leading zero, tie case
    # 4. Three leading zeros, round up
    # 5. No leading zeros, no rounding needed
    inputs = {
        'abs_mantissa': [0b010000000,  # Normal case
                        0b001100000,  # 2 leading zeros
                        0b001000000,  # 2 leading zeros
                        0b000100000,  # 3 leading zeros
                        0b010000000], # No rounding
        'sticky':       [1, 0, 0, 1, 0],
        'guard':        [1, 0, 1, 1, 0],
        'round':        [0, 0, 0, 0, 0],
        'lzc':          [0, 2, 1, 3, 1]
    }
    
    sim.step_multiple(inputs)
    
    input_repr_map = {
        'abs_mantissa': repr_mantissa_sum,
        'sticky': repr_num,
        'guard': repr_num,
        'round': repr_num,
        'lzc': repr_num,
        'norm_shift': repr_mantissa_sum,
        'rounded_mantissa': repr_mantissa_sum,
        'final_mantissa': repr_mantissa_hidden,
        'inc_out': repr_num
    }
    
    trace_list = list(input_repr_map.keys())
    custom_render_trace(sim_trace, trace_list=trace_list, repr_per_name=input_repr_map)

def test_adjust_final_exponent():
    reset_working_block()
    
    # Create inputs
    exp_larger = Input(E_BITS, 'exp_larger')
    lzc = Input(4, 'lzc')
    round_inc = Input(1, 'round_inc')
    
    # Get output
    final_exp = adjust_final_exponent(exp_larger, lzc, round_inc, E_BITS)
    
    # Create output wire
    exp_out = Output(E_BITS, 'exp_out')
    exp_out <<= final_exp
    
    # Simulate
    sim_trace = SimulationTrace()
    sim = Simulation(tracer=sim_trace)
    
    # Test cases:
    # 1. Normal case, no round increment
    # 2. Normal case with round increment
    # 3. Large LZC value
    # 4. Near maximum exponent
    # 5. Near minimum exponent
    inputs = {
        'exp_larger': [128,  128,  130,  254,  3],
        'lzc':        [2,    2,    5,    0,    2],
        'round_inc':  [0,    1,    0,    1,    0]
    }
    
    sim.step_multiple(inputs)
    
    input_repr_map = {
        'exp_larger': repr_exp,
        'lzc': repr_num,
        'round_inc': repr_num,
        'lzc_adjusted': repr_exp,
        'exp_out': repr_exp
    }
    
    trace_list = list(input_repr_map.keys())
    custom_render_trace(sim_trace, trace_list=trace_list, repr_per_name=input_repr_map)

