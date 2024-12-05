# from inside q1_project/src
# python -m output_generators.make_svgs

import numpy as np
import pyrtl
from utils import create_svg
from bf16_adder import PipelinedBF16Adder, bfloat16_adder_combinatorial
from bf16_multiplier import PipelinedBF16Multiplier
from basic_systolic import SystolicMacArray
from lmul import *
import os

OUTPUT_PATH = "output"

if __name__=="__main__":
    rtl_funcs = [
        bf16_lmul_naive,
        bf16_lmul_combinatorial,
        fp8_lmul_combinatorial,
    ]
    for rtl_func in rtl_funcs:
        create_svg(rtl_func, output_path=OUTPUT_PATH)

    create_svg(FastPipelinedLMULFP8, output_path=OUTPUT_PATH, split_state=False)
    create_svg(PipelinedBF16Multiplier, output_path=OUTPUT_PATH, split_state=False)


    pyrtl.reset_working_block()
    PipelinedBF16Adder(pyrtl.Input(16, 'a'), pyrtl.Input(16, 'b'), pyrtl.Input(1, 'write_enable'))
    with open(os.path.join(OUTPUT_PATH, "PipelinedBF16Adder.svg"), "w") as f:
        pyrtl.output_to_svg(f, split_state=False)
    
    pyrtl.reset_working_block()
    bfloat16_adder_combinatorial(pyrtl.Input(16, 'a'), pyrtl.Input(16, 'b'))
    with open(os.path.join(OUTPUT_PATH, "bf16_combinatorial_adder.svg"), "w") as f:
        pyrtl.output_to_svg(f, split_state=False)

    pyrtl.reset_working_block()
    x = np.array([[1,2,3],[4,5,6],[7,8,9]])
    a = np.array([[2,0,0],[0,4,0],[0,0,8]])
    systolic_sim = SystolicMacArray(x, a, 3)
    with open(os.path.join(OUTPUT_PATH, "basic_systolic_array.svg"), "w") as f:
        pyrtl.output_to_svg(f, split_state=False)
    