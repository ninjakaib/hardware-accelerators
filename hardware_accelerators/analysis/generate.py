from pathlib import Path
from h11 import Data
from pandas import DataFrame
import pyrtl
from itertools import product
from pyrtl import *
from dataclasses import dataclass
from typing import Callable, Type, Literal, Optional

from .verilog_export import export_to_verilog
from ..dtypes import *
from ..rtllib import *
from ..rtllib.processing_element import ProcessingElement
from ..rtllib.adders import *
from ..rtllib.multipliers import *
from ..rtllib.lmul import *
from ..rtllib.utils.common import *
from ..simulation.utils import *


def create_inputs(**named_bitwidths):
    """
    Create PyRTL Input wires with specified bitwidths.

    Args:
        **named_bitwidths: Named bitwidths where the key is used as the wire name

    Returns:
        Generator of PyRTL Input wires

    Note:
        You must use all keyword arguments
    """

    # If using keyword arguments
    for name, bitwidth in named_bitwidths.items():
        yield pyrtl.Input(bitwidth, name=name)  # type: ignore


def create_outputs(*args, **named_wires):
    """
    Create PyRTL Output wires connected to the input wires.

    Args:
        *args: Variable number of wires to connect to unnamed outputs
        **named_wires: Named wires where the key is used as the output wire name

    Note:
        You must use either all positional arguments or all keyword arguments, not a mix.
    """
    if args and named_wires:
        raise ValueError(
            "Please use either all positional arguments or all keyword arguments, not a mix."
        )

    # If using positional arguments
    for wire in args:
        out = pyrtl.Output(len(wire), name=wire.name.replace("tmp", "out"))  # type: ignore
        out <<= wire

    # If using keyword arguments
    for name, wire in named_wires.items():
        out = pyrtl.Output(len(wire), name=name)  # type: ignore
        out <<= wire


@dataclass
class RTLAnalysis:
    """Results of RTL analysis."""

    max_delay: float
    max_freq: float
    logic_area: float
    mem_area: float
    name: Optional[str] = None

    def __repr__(self):
        if self.name is None:
            return (
                f"RTLAnalysisResults("
                f"max_delay={self.max_delay:.2f} ps, "
                f"max_freq={self.max_freq:.2f} MHz, "
                f"logic_area={self.logic_area:.2f}um², "
                f"mem_area={self.mem_area:.2f}um²)"
            )
        else:
            return (
                f"RTLAnalysisResults for {self.name}:\n\t"
                f"max_delay={self.max_delay:.2f} ps\n\t"
                f"max_freq={self.max_freq:.2f} MHz\n\t"
                f"logic_area={self.logic_area:.2f}um²\n\t"
                f"mem_area={self.mem_area:.2f}um²"
            )


def analyze(
    block: Block | None = None, synth: bool = True, opt: bool = True, name=None
):
    if block is not None:
        pyrtl.set_working_block(block)

    if synth:
        pyrtl.synthesize()
    if opt:
        pyrtl.optimize()

    timing = pyrtl.TimingAnalysis()
    max_delay = timing.max_length()
    max_freq = timing.max_freq()
    logic_area, mem_area = pyrtl.area_estimation()

    return RTLAnalysis(
        name=name,
        max_delay=max_delay,
        max_freq=max_freq,
        logic_area=logic_area * 1e6,
        mem_area=mem_area * 1e6,
    )


def create_adder_blocks(dtype: Type[BaseFloat]) -> dict[str, Block]:
    bits = dtype.bitwidth()
    e_bits, m_bits = dtype.exponent_bits(), dtype.mantissa_bits()

    combinational_block = pyrtl.Block()
    combinational_fast_block = pyrtl.Block()
    adder_pipelined_block = pyrtl.Block()
    adder_pipelined_fast_block = pyrtl.Block()
    stage_2_block = pyrtl.Block()
    stage_2_fast_block = pyrtl.Block()
    stage_3_block = pyrtl.Block()
    stage_4_block = pyrtl.Block()
    stage_4_fast_block = pyrtl.Block()
    stage_5_block = pyrtl.Block()

    # Combinational design
    with set_working_block(combinational_block):
        create_outputs(
            *float_adder(
                *create_inputs(float_a=bits, float_b=bits), dtype=dtype, fast=False
            )
        )

    with set_working_block(combinational_fast_block):
        create_outputs(
            *float_adder(
                *create_inputs(float_a=bits, float_b=bits), dtype=dtype, fast=True
            )
        )

    # Complete pipelined design
    with set_working_block(adder_pipelined_block):
        create_outputs(
            float_adder_pipelined(
                *create_inputs(float_a=bits, float_b=bits),
                dtype=dtype,
                fast=False,
            )
        )

    with set_working_block(adder_pipelined_fast_block):
        create_outputs(
            float_adder_pipelined(
                *create_inputs(float_a=bits, float_b=bits),
                dtype=dtype,
                fast=True,
            )
        )

    # Stages 1 & 2
    with set_working_block(stage_2_block):
        float_components = extract_float_components(
            *create_inputs(float_a=bits, float_b=bits),
            e_bits=e_bits,
            m_bits=m_bits,
        )
        stage_2_outputs = adder_stage_2(
            *float_components,
            e_bits,
            m_bits,
            fast=False,
        )
        create_outputs(*stage_2_outputs)

    with set_working_block(stage_2_fast_block):
        float_components = extract_float_components(
            *create_inputs(float_a=bits, float_b=bits),
            e_bits=e_bits,
            m_bits=m_bits,
        )
        stage_2_outputs = adder_stage_2(
            *float_components,
            e_bits,
            m_bits,
            fast=True,
        )
        create_outputs(*stage_2_outputs)

    # Stage 3
    with set_working_block(stage_3_block):
        # Perform alignment and generate SGR bits
        stage_3_outputs = adder_stage_3(
            *create_inputs(mant_smaller=m_bits + 1, shift_amount=e_bits),
            e_bits=e_bits,
            m_bits=m_bits,
        )
        create_outputs(*stage_3_outputs)

    # Stage 4
    with set_working_block(stage_4_block):
        # Perform mantissa addition and leading zero detection
        stage_4_outputs = adder_stage_4(
            *create_inputs(mant_aligned=m_bits + 1, mant_unchanged=m_bits + 1, s_xor=1),
            m_bits=m_bits,
            fast=False,
        )
        create_outputs(*stage_4_outputs)

    with set_working_block(stage_4_fast_block):
        # Perform mantissa addition and leading zero detection
        stage_4_outputs = adder_stage_4(
            *create_inputs(mant_aligned=m_bits + 1, mant_unchanged=m_bits + 1, s_xor=1),
            m_bits=m_bits,
            fast=True,
        )
        create_outputs(*stage_4_outputs)

    # Stage 5
    with set_working_block(stage_5_block):
        # Perform normalization, rounding, and final assembly
        stage_5_outputs = adder_stage_5(
            *create_inputs(
                abs_mantissa=m_bits + 2,
                sticky_bit=1,
                guard_bit=1,
                round_bit=1,
                lzc=4,
                exp_larger=e_bits,
                sign_a=1,
                sign_b=1,
                exp_diff=e_bits + 1,
                is_neg=1,
            ),
            e_bits=e_bits,
            m_bits=m_bits,
        )
        create_outputs(*stage_5_outputs)

    # Return all the generated blocks for analysis
    return {
        "adder_combinational": combinational_block,
        "adder_combinational_fast": combinational_fast_block,
        "adder_pipelined": adder_pipelined_block,
        "adder_pipelined_fast": adder_pipelined_fast_block,
        "adder_stage_2": stage_2_block,
        "adder_stage_2_fast": stage_2_fast_block,
        "adder_stage_3": stage_3_block,
        "adder_stage_4": stage_4_block,
        "adder_stage_4_fast": stage_4_fast_block,
        "adder_stage_5": stage_5_block,
    }


def create_multiplier_blocks(dtype: Type[BaseFloat], fast: bool) -> dict[str, Block]:
    bits = dtype.bitwidth()
    e_bits, m_bits = dtype.exponent_bits(), dtype.mantissa_bits()

    combinational_block = pyrtl.Block()
    multiplier_block = pyrtl.Block()
    stage_2_block = pyrtl.Block()
    stage_3_block = pyrtl.Block()
    stage_4_block = pyrtl.Block()

    # Combinational design
    with set_working_block(combinational_block):
        create_outputs(
            float_multiplier(
                *create_inputs(float_a=bits, float_b=bits), dtype=dtype, fast=fast
            )
        )

    # Complete pipelined design
    with set_working_block(multiplier_block):
        multiplier = FloatMultiplierPipelined(
            *create_inputs(float_a=bits, float_b=bits), dtype=dtype, fast=fast
        )
        create_outputs(multiplier._result)

    # Stage 1 & 2: Extract components and calculate sign, exponent sum, mantissa product
    with set_working_block(stage_2_block):
        float_components = extract_float_components(
            *create_inputs(float_a=bits, float_b=bits),
            e_bits=e_bits,
            m_bits=m_bits,
        )
        stage_2_outputs = multiplier_stage_2(
            *float_components,
            m_bits,
            fast,
        )
        create_outputs(*stage_2_outputs)

    # Stage 3: Leading zero detection and exponent adjustment
    with set_working_block(stage_3_block):
        stage_3_outputs = multiplier_stage_3(
            *create_inputs(exp_sum=e_bits + 1, mant_product=2 * m_bits + 2),
            e_bits=e_bits,
            m_bits=m_bits,
            fast=fast,
        )
        create_outputs(*stage_3_outputs)

    # Stage 4: Normalization, rounding, and final assembly
    with set_working_block(stage_4_block):
        stage_4_outputs = multiplier_stage_4(
            *create_inputs(
                unbiased_exp=e_bits,
                leading_zeros=e_bits,
                mantissa_product=2 * m_bits + 2,
            ),
            m_bits=m_bits,
            e_bits=e_bits,
            fast=fast,
        )
        create_outputs(*stage_4_outputs)

    # Return all the generated blocks for analysis
    faststr = "_fast" if fast else ""
    return {
        f"multiplier_combinational{faststr}": combinational_block,
        f"multiplier{faststr}": multiplier_block,
        f"multiplier_stage_2{faststr}": stage_2_block,
        f"multiplier_stage_3{faststr}": stage_3_block,
        f"multiplier_stage_4{faststr}": stage_4_block,
    }


def create_lmul_blocks(dtype: Type[BaseFloat]) -> dict[str, Block]:
    bits = dtype.bitwidth()

    combinational_block = pyrtl.Block()
    combinational_fast_block = pyrtl.Block()
    pipelined_block = pyrtl.Block()
    pipelined_fast_block = pyrtl.Block()

    # Combinational design (simple)
    with set_working_block(combinational_block):
        create_outputs(
            lmul_simple(*create_inputs(float_a=bits, float_b=bits), dtype=dtype)
        )

    # Combinational design (fast)
    with set_working_block(combinational_fast_block):
        create_outputs(
            lmul_fast(*create_inputs(float_a=bits, float_b=bits), dtype=dtype)
        )

    # Pipelined design (simple)
    with set_working_block(pipelined_block):
        mult = LmulPipelined(
            *create_inputs(float_a=bits, float_b=bits), dtype=dtype, fast=False
        )
        create_outputs(mult.output_reg)

    # Pipelined design (fast)
    with set_working_block(pipelined_fast_block):
        mult = LmulPipelined(
            *create_inputs(float_a=bits, float_b=bits), dtype=dtype, fast=True
        )
        create_outputs(mult.output_reg)

    # Return all the generated blocks for analysis
    return {
        "lmul_combinational_simple": combinational_block,
        "lmul_combinational_fast": combinational_fast_block,
        "lmul_pipelined_simple": pipelined_block,
        "lmul_pipelined_fast": pipelined_fast_block,
    }


def connect_pe_io(pe: ProcessingElement):
    # Connect the inputs and outputs of the processing element
    w_bits, a_bits = pe.weight_type.bitwidth(), pe.data_type.bitwidth()
    w_in, d_in, acc_in = create_inputs(
        weight_in=w_bits, data_in=a_bits, accum_in=a_bits
    )
    pe.connect_weight(w_in)
    pe.connect_data(d_in)
    pe.connect_accum(acc_in)
    pe.connect_control_signals(
        *create_inputs(weight_en=1, data_en=1, mul_en=1, adder_en=1)
    )
    create_outputs(*pe.outputs.__dict__.values())


def create_pe_blocks(
    dtypes: tuple[Type[BaseFloat], Type[BaseFloat]],
) -> dict[str, Block]:
    """Create a processing element for each pair of dtypes."""

    weight_dtype, act_dtype = dtypes

    # Defining blocks to encapsulate hardware

    combinational_block = Block()
    simple_pipeline_block = Block()
    simple_pipeline_fast_block = Block()
    full_pipeline_block = Block()
    full_pipeline_fast_block = Block()

    combinational_lmul_block = Block()
    simple_pipeline_lmul_block = Block()
    simple_pipeline_fast_lmul_block = Block()
    full_pipeline_lmul_block = Block()
    full_pipeline_fast_lmul_block = Block()

    # Standard IEEE multiplier versions

    with set_working_block(combinational_block):
        pe = ProcessingElement(
            data_type=act_dtype,
            weight_type=weight_dtype,
            accum_type=act_dtype,
            multiplier=float_multiplier,
            adder=float_adder,
            pipeline_mult=False,
        )
        connect_pe_io(pe)

    with set_working_block(simple_pipeline_block):
        pe = ProcessingElement(
            data_type=act_dtype,
            weight_type=weight_dtype,
            accum_type=act_dtype,
            multiplier=float_multiplier,
            adder=float_adder,
            pipeline_mult=True,
        )
        connect_pe_io(pe)

    with set_working_block(simple_pipeline_fast_block):
        pe = ProcessingElement(
            data_type=act_dtype,
            weight_type=weight_dtype,
            accum_type=act_dtype,
            multiplier=float_multiplier_fast_unstable,
            adder=float_adder_fast_unstable,
            pipeline_mult=True,
        )
        connect_pe_io(pe)

    with set_working_block(full_pipeline_block):
        pe = ProcessingElement(
            data_type=act_dtype,
            weight_type=weight_dtype,
            accum_type=act_dtype,
            multiplier=float_multiplier_pipelined,
            adder=float_adder_pipelined,
            pipeline_mult=True,
        )
        connect_pe_io(pe)

    with set_working_block(full_pipeline_fast_block):
        pe = ProcessingElement(
            data_type=act_dtype,
            weight_type=weight_dtype,
            accum_type=act_dtype,
            multiplier=float_multiplier_pipelined_fast_unstable,
            adder=float_adder_pipelined_fast_unstable,
            pipeline_mult=True,
        )
        connect_pe_io(pe)

    # L-mul versions

    with set_working_block(combinational_lmul_block):
        pe = ProcessingElement(
            data_type=act_dtype,
            weight_type=weight_dtype,
            accum_type=act_dtype,
            multiplier=lmul_simple,
            adder=float_adder,
            pipeline_mult=False,
        )
        connect_pe_io(pe)

    with set_working_block(simple_pipeline_lmul_block):
        pe = ProcessingElement(
            data_type=act_dtype,
            weight_type=weight_dtype,
            accum_type=act_dtype,
            multiplier=lmul_simple,
            adder=float_adder,
            pipeline_mult=True,
        )
        connect_pe_io(pe)

    with set_working_block(simple_pipeline_fast_lmul_block):
        pe = ProcessingElement(
            data_type=act_dtype,
            weight_type=weight_dtype,
            accum_type=act_dtype,
            multiplier=lmul_fast,
            adder=float_adder_fast_unstable,
            pipeline_mult=True,
        )
        connect_pe_io(pe)

    with set_working_block(full_pipeline_lmul_block):
        pe = ProcessingElement(
            data_type=act_dtype,
            weight_type=weight_dtype,
            accum_type=act_dtype,
            multiplier=lmul_pipelined,
            adder=float_adder_pipelined,
            pipeline_mult=True,
        )
        connect_pe_io(pe)

    with set_working_block(full_pipeline_fast_lmul_block):
        pe = ProcessingElement(
            data_type=act_dtype,
            weight_type=weight_dtype,
            accum_type=act_dtype,
            multiplier=lmul_pipelined_fast,
            adder=float_adder_pipelined_fast_unstable,
            pipeline_mult=True,
        )
        connect_pe_io(pe)

    return {
        "pe_combinational": combinational_block,
        "pe_standard": simple_pipeline_block,
        "pe_fast": simple_pipeline_fast_block,
        "pe_pipelined": full_pipeline_block,
        "pe_fast_pipelined": full_pipeline_fast_block,
        "pe_combinational_lmul": combinational_lmul_block,
        "pe_standard_lmul": simple_pipeline_lmul_block,
        "pe_fast_lmul": simple_pipeline_fast_lmul_block,
        "pe_pipelined_lmul": full_pipeline_lmul_block,
        "pe_fast_pipelined_lmul": full_pipeline_fast_lmul_block,
    }


def create_accelerator_blocks(
    dtypes: tuple[Type[BaseFloat], Type[BaseFloat]],
    array_size: int = 4,
    addr_bits: int = 12,
) -> dict[str, Block]:
    """
    Create accelerator blocks for all valid configurations based on the given inputs.

    Args:
        dtypes: Tuple of (weight_type, activation_type) data types
        array_size: Size of the systolic array (N x N)
        addr_bits: Bit width for accumulator address (uses default if None)

    Returns:
        Dictionary mapping configuration names to PyRTL blocks
    """
    weight_type, activation_type = dtypes

    # Define all valid configurations to test
    pipeline_options = [None, "low", "high"]
    lmul_options = [False, True]
    fast_options = [False, True]

    # Create configs and blocks
    blocks = {}
    for pipeline, lmul, fast in product(pipeline_options, lmul_options, fast_options):
        if pipeline is None and fast is True:
            continue

        # Create the configuration
        config = AcceleratorAnalysisConfig(
            array_size=array_size,
            activation_type=activation_type,
            weight_type=weight_type,
            lmul=lmul,
            accum_addr_width=addr_bits,
            pipeline_level=pipeline,
            use_fast_internals=fast,
        )

        block = pyrtl.Block()
        with set_working_block(block):
            AcceleratorTopLevel(config)

        blocks[config.name] = block

    return blocks


################################################################

# if __name__ == "__main__":

#     OUTPUT_DIR = Path("verilog")
#     POSTSYNTH_DIR = OUTPUT_DIR / "pyrtl_synth"

#     EXPORT_PRE_SYNTH = False
#     EXPORT_POST_SYNTH = True
#     RUN_ANALYSIS = True
#     ANALYSIS_RESULT_DIR = Path("results")

#     array_size = 8
#     addr_bits = 12

#     dtype_list = [Float8, BF16, Float32]

#     dtype_names = {Float8: "fp8", BF16: "bf16", Float32: "fp32"}

#     weight_act_dtypes = [
#         (Float8, Float8),
#         (Float8, BF16),
#         (Float8, Float32),
#         (BF16, BF16),
#         (BF16, Float32),
#         (Float32, Float32),
#     ]

#     # Hardware building blocks
#     basic_component_analysis = []

#     for dtype in dtype_list:
#         block_dicts = [
#             ("adder", create_adder_blocks(dtype)),
#             ("multiplier", create_multiplier_blocks(dtype, fast=False)),
#             ("multiplier", create_multiplier_blocks(dtype, fast=True)),
#             ("lmul", create_lmul_blocks(dtype)),
#         ]
#         for component_name, block_dict in block_dicts:
#             for name, block in block_dict.items():
#                 output_path = Path(component_name, dtype_names[dtype], f"{name}.v")
#                 if EXPORT_PRE_SYNTH:
#                     export_to_verilog(block, OUTPUT_DIR / output_path)
#                 if RUN_ANALYSIS:
#                     analysis_result = analyze(block, name=name)
#                     analysis_result.dtype = dtype_names[dtype]
#                     analysis_result.component = component_name
#                     basic_component_analysis.append(analysis_result.__dict__)
#                     if EXPORT_POST_SYNTH:
#                         export_to_verilog(block, POSTSYNTH_DIR / output_path)

#     # More complex hardware
#     pe_analysis = []
#     accelerator_analysis = []

#     for weight_dtype, act_dtype in weight_act_dtypes:
#         folder_name = f"w{weight_dtype.bitwidth()}a{act_dtype.bitwidth()}"

#         pe_blocks = create_pe_blocks((weight_dtype, act_dtype))
#         for name, block in pe_blocks.items():
#             pe_output_path = Path("pe", folder_name, f"{name}.v")
#             if EXPORT_PRE_SYNTH:
#                 export_to_verilog(block, OUTPUT_DIR / pe_output_path)
#             if RUN_ANALYSIS:
#                 analysis_result = analyze(block, name=name)
#                 analysis_result.weights = dtype_names[weight_dtype]
#                 analysis_result.activations = dtype_names[act_dtype]
#                 analysis_result.component = "pe"
#                 pe_analysis.append(analysis_result.__dict__)
#                 if EXPORT_POST_SYNTH:
#                     export_to_verilog(block, POSTSYNTH_DIR / pe_output_path)

#         accelerator_blocks = create_accelerator_blocks(
#             (weight_dtype, act_dtype), array_size, addr_bits
#         )
#         for name, block in accelerator_blocks.items():
#             accelerator_output_path = Path("accelerator", folder_name, f"{name}.v")
#             if EXPORT_PRE_SYNTH:
#                 export_to_verilog(block, OUTPUT_DIR / accelerator_output_path)
#             if RUN_ANALYSIS:
#                 analysis_result = analyze(block, name=name)
#                 analysis_result.weights = dtype_names[weight_dtype]
#                 analysis_result.activations = dtype_names[act_dtype]
#                 analysis_result.component = "accelerator"
#                 accelerator_analysis.append(analysis_result.__dict__)
#                 if EXPORT_POST_SYNTH:
#                     export_to_verilog(block, POSTSYNTH_DIR / accelerator_output_path)

#     if RUN_ANALYSIS:
#         DataFrame(basic_component_analysis).to_csv(
#             ANALYSIS_RESULT_DIR / "component_analysis.csv", index=False
#         )
#         DataFrame(pe_analysis).to_csv(
#             ANALYSIS_RESULT_DIR / "pe_analysis.csv", index=False
#         )
#         DataFrame(accelerator_analysis).to_csv(
#             ANALYSIS_RESULT_DIR / "accelerator_analysis.csv", index=False
#         )


import multiprocessing as mp
from pathlib import Path
import os
import csv
import time
from functools import partial
from pandas import DataFrame
import json
import traceback


def process_block(
    block,
    name,
    output_dir,
    postsynth_dir,
    export_pre_synth,
    export_post_synth,
    run_analysis,
    analysis_result_dir,
    component_name=None,
    dtype=None,
    weight_dtype=None,
    act_dtype=None,
    dtype_names=None,
    output_path=None,
):
    """Process a single block with optional export and analysis"""
    result = None
    try:
        if export_pre_synth and output_path:
            os.makedirs((output_dir / output_path).parent, exist_ok=True)
            export_to_verilog(block, output_dir / output_path)

        if run_analysis:
            analysis_result = analyze(block, name=name)

            # Set appropriate attributes based on the component type
            if component_name and dtype:
                analysis_result.dtype = dtype_names[dtype]
                analysis_result.component = component_name
                result_type = "component"
            elif weight_dtype and act_dtype:
                analysis_result.weights = dtype_names[weight_dtype]
                analysis_result.activations = dtype_names[act_dtype]
                analysis_result.component = component_name
                result_type = "pe" if component_name == "pe" else "accelerator"

            result = (result_type, analysis_result.__dict__)

            if export_post_synth and output_path:
                os.makedirs((postsynth_dir / output_path).parent, exist_ok=True)
                export_to_verilog(block, postsynth_dir / output_path)

        return result
    except Exception as e:
        error_msg = f"Error processing {name}: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        with open(analysis_result_dir / "errors.log", "a") as f:
            f.write(f"{error_msg}\n{'='*80}\n")
        return None


def save_result(result, analysis_result_dir, result_files):
    """Save a single result to the appropriate CSV file"""
    if not result:
        return

    result_type, data = result

    # Get the appropriate file path and headers
    if result_type == "component":
        file_path = analysis_result_dir / "component_analysis.csv"
    elif result_type == "pe":
        file_path = analysis_result_dir / "pe_analysis.csv"
    elif result_type == "accelerator":
        file_path = analysis_result_dir / "accelerator_analysis.csv"

    # Create directory if it doesn't exist
    os.makedirs(file_path.parent, exist_ok=True)

    # Check if file exists to determine if we need to write headers
    file_exists = file_path.exists()

    # Get headers from the data
    headers = list(data.keys())

    # Open file in append mode
    with open(file_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)

        # Write headers if file doesn't exist
        if not file_exists:
            writer.writeheader()

        # Write the data row
        writer.writerow(data)

    # Track that we've written to this file
    result_files.add(file_path)


def result_callback(result, analysis_result_dir, result_files):
    """Callback function for when a process completes"""
    if result:
        save_result(result, analysis_result_dir, result_files)


if __name__ == "__main__":
    # Create a multiprocessing pool with as many processes as CPU cores
    pool = mp.Pool(processes=mp.cpu_count())

    # Set to track which result files we've written to
    result_files = set()

    OUTPUT_DIR = Path("verilog")
    POSTSYNTH_DIR = OUTPUT_DIR / "pyrtl_synth"

    EXPORT_PRE_SYNTH = False
    EXPORT_POST_SYNTH = True
    RUN_ANALYSIS = True
    ANALYSIS_RESULT_DIR = Path("results")

    # Create output directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(POSTSYNTH_DIR, exist_ok=True)
    os.makedirs(ANALYSIS_RESULT_DIR, exist_ok=True)

    array_size = 8
    addr_bits = 12

    dtype_list = [Float8, BF16, Float32]

    dtype_names = {Float8: "fp8", BF16: "bf16", Float32: "fp32"}

    weight_act_dtypes = [
        (Float8, Float8),
        (Float8, BF16),
        (Float8, Float32),
        (BF16, BF16),
        (BF16, Float32),
        (Float32, Float32),
    ]

    # Create a partial function with common arguments
    process_block_partial = partial(
        process_block,
        output_dir=OUTPUT_DIR,
        postsynth_dir=POSTSYNTH_DIR,
        export_pre_synth=EXPORT_PRE_SYNTH,
        export_post_synth=EXPORT_POST_SYNTH,
        run_analysis=RUN_ANALYSIS,
        analysis_result_dir=ANALYSIS_RESULT_DIR,
        dtype_names=dtype_names,
    )

    # Create a callback function with common arguments
    callback = partial(
        result_callback,
        analysis_result_dir=ANALYSIS_RESULT_DIR,
        result_files=result_files,
    )

    # Track all submitted tasks
    tasks = []

    # Hardware building blocks
    for dtype in dtype_list:
        block_dicts = [
            ("adder", create_adder_blocks(dtype)),
            ("multiplier", create_multiplier_blocks(dtype, fast=False)),
            ("multiplier", create_multiplier_blocks(dtype, fast=True)),
            ("lmul", create_lmul_blocks(dtype)),
        ]

        for component_name, block_dict in block_dicts:
            for name, block in block_dict.items():
                output_path = Path(component_name, dtype_names[dtype], f"{name}.v")

                # Submit task to process pool
                task = pool.apply_async(
                    process_block_partial,
                    args=(block, name),
                    kwds={
                        "component_name": component_name,
                        "dtype": dtype,
                        "output_path": output_path,
                    },
                    callback=callback,
                )
                tasks.append(task)

    # More complex hardware
    for weight_dtype, act_dtype in weight_act_dtypes:
        folder_name = f"w{weight_dtype.bitwidth()}a{act_dtype.bitwidth()}"

        # Process PE blocks
        pe_blocks = create_pe_blocks((weight_dtype, act_dtype))
        for name, block in pe_blocks.items():
            pe_output_path = Path("pe", folder_name, f"{name}.v")

            task = pool.apply_async(
                process_block_partial,
                args=(block, name),
                kwds={
                    "component_name": "pe",
                    "weight_dtype": weight_dtype,
                    "act_dtype": act_dtype,
                    "output_path": pe_output_path,
                },
                callback=callback,
            )
            tasks.append(task)

        # Process accelerator blocks
        accelerator_blocks = create_accelerator_blocks(
            (weight_dtype, act_dtype), array_size, addr_bits
        )
        for name, block in accelerator_blocks.items():
            accelerator_output_path = Path("accelerator", folder_name, f"{name}.v")

            task = pool.apply_async(
                process_block_partial,
                args=(block, name),
                kwds={
                    "component_name": "accelerator",
                    "weight_dtype": weight_dtype,
                    "act_dtype": act_dtype,
                    "output_path": accelerator_output_path,
                },
                callback=callback,
            )
            tasks.append(task)

    # Wait for all tasks to complete
    try:
        # Monitor progress
        total_tasks = len(tasks)
        completed = 0
        print(f"Processing {total_tasks} tasks using {mp.cpu_count()} processes")

        while completed < total_tasks:
            new_completed = sum(1 for task in tasks if task.ready())
            if new_completed > completed:
                completed = new_completed
                print(
                    f"Progress: {completed}/{total_tasks} tasks completed ({completed/total_tasks*100:.1f}%)"
                )
            time.sleep(1)

        # Make sure all tasks are properly completed
        for task in tasks:
            task.get()  # This will raise any exceptions that occurred in the task

    except KeyboardInterrupt:
        print("Process interrupted by user. Partial results have been saved.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Partial results have been saved.")
    finally:
        # Close the pool
        pool.close()
        pool.join()

        print(f"Results saved to: {', '.join(str(f) for f in result_files)}")
