# Run with:
# cd q1_project/src
# python3 -m output_generators.bf16_lmul_error_heatmap

from bfloat16 import BF16
from lmul import bf16_lmul_naive
from typing import Any, Callable
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from itertools import combinations_with_replacement
import pyrtl
import pyrtl.simulation

OUTPUT_PATH = "output/bf16_lmul_error_heatmap.png"

def get_bf16_special_values():
    return {
        # Zeros
        "+Zero": BF16(0.0),     
        "-Zero": BF16(-0.0),    
        # Infinities
        "+Infinity": BF16(float('inf')),    
        "-Infinity": BF16(float('-inf')),   
        # NaN
        "NaN": BF16(float('nan')),    
        # Normal edge cases
        "+MaxNormal": BF16('0.11111110.1111111'), 
        "-MaxNormal": BF16('1.11111110.1111111'), 
        "+MinNormal": BF16('0.00000001.0000000'), 
        "-MinNormal": BF16('1.00000001.0000000'), 
        # Subnormal edge cases
        "+MaxSubnormal": BF16('0.00000000.1111111'), 
        "-MaxSubnormal": BF16('1.00000000.1111111'), 
        "+MinSubnormal": BF16('0.00000000.0000001'), 
        "-MinSubnormal": BF16('1.00000000.0000001'), 
        # Regular normal number
        "+One": BF16(1.0),       
        "-One": BF16(-1.0),      
    }

sv = get_bf16_special_values()
sv_arr = np.array(list(sv.values()))

result_matrix = sv_arr.reshape(-1,1) @ sv_arr.reshape(1,-1)

result_matrix_df = pd.DataFrame(result_matrix, index=sv.keys(), columns=sv.keys())
result_matrix_df

def create_bf16_lmul_testbench(
    input_names: list[str], 
    output_name: str,
    start: int = -3,
    stop: int = 0,
    n_values: int = None,
    include_special_values: bool = True,
    pipeline_stages: int = 0
) -> tuple[dict[str, list[int]]]:
    # Assert input names contains exactly 2 elements
    assert len(input_names) == 2, "Input names must contain exactly 2 elements"
    if n_values is None:
        n_values = abs(stop - start) + 1

    # Generate values in a given range, plus some special values to represent edge cases
    values = [BF16(i) for i in np.logspace(start, stop, num=n_values)]
    special_values = list(get_bf16_special_values().values())
    
    # Generate all combinations
    pairs = list(combinations_with_replacement(values, 2))
    if include_special_values:
        pairs += list(combinations_with_replacement(special_values, 2))

    # Transpose the array and calculate the products elementwise
    bf16_inputs = np.array(pairs).T
    bf16_outputs = bf16_inputs[0] * bf16_inputs[1]

    bits: Callable[[BF16], Any] = lambda x: x.bit_representation()
    pipeline_buffer = [0]*pipeline_stages

    raw_inputs = {name: list(map(bits, input)) + pipeline_buffer for name, input in zip(input_names, bf16_inputs)}
    expected_outputs = {output_name: pipeline_buffer + list(map(bits, bf16_outputs))}

    return {
        "inputs": raw_inputs,
        "outputs": expected_outputs,
    }

def run_testbench(
    rtl_func,
    input_names: list[str], 
    output_name: str,
    start: int = -3,
    stop: int = 0,
    n_values: int = None,
    include_special_values: bool = True,
    pipeline_stages: int = 0
):
    test_cases = create_bf16_lmul_testbench(
        input_names,
        output_name,
        start,
        stop,
        n_values,
        include_special_values,
        pipeline_stages
    )
    inputs = test_cases["inputs"]
    outputs = test_cases["outputs"]

    pyrtl.reset_working_block()
    rtl_func()

    n_steps = len(sorted(list(inputs.values()), key=lambda x: len(x), reverse=True)[0])

    sim_trace = pyrtl.SimulationTrace()
    sim = pyrtl.Simulation(tracer=sim_trace)
    results = []
    for i in range(n_steps):
        sim.step({w: int(v[i]) for w, v in inputs.items()})
        results.append(sim.inspect(output_name))

    renamed_inputs = {f'input{i+1}': inputs[input_names[i]] for i in range(len(input_names))}
    results_df = pd.DataFrame(data={**renamed_inputs, 'expected': outputs[output_name], 'actual': results})
    convert_int_to_bf16 = lambda x: float(BF16.from_binary(x))
    results_df = results_df.map(convert_int_to_bf16)
    results_df['diff'] = results_df['actual'] - results_df['expected']
    return results_df


def reshape_testbench_results(df: pd.DataFrame):
    # return df.pivot(
    #     index='input1',
    #     columns='input2',
    #     values='diff'
    # )
    pivot_df = df.pivot_table(
        index='input2', 
        columns='input1', 
        values='diff', 
        # aggfunc='first'
    )
    return pivot_df[::-1]

def create_diff_heatmap_a(pivot_df):
    """
    Create a heatmap visualization of the diff values using a symmetric log color scale.
    
    Parameters:
    -----------
    pivot_df : pandas.DataFrame
        Reshaped DataFrame with input1 as index and input2 as columns
    
    Returns:
    --------
    matplotlib.figure.Figure
        Heatmap visualization
    """
    # Create a symmetric log normalization
    # This handles both positive and negative values on a log scale
    linthresh = 1e-7  # Linear threshold around zero
    norm = mcolors.SymLogNorm(
        linthresh=linthresh,
        linscale=1,
        vmin=pivot_df.values.min(),
        vmax=pivot_df.values.max(),
        base=10
    )
    
    # Create figure and axes
    plt.figure(figsize=(12, 10))
    
    # Create heatmap using seaborn with custom normalization
    sns.heatmap(
        pivot_df, 
        cmap='coolwarm',  # Red-Blue color map
        norm=norm,        # Symmetric log normalization
        # center=0,         # Center colormap at 0
        # annot=True,       # Show numeric values in each cell
        fmt='.2e',        # Scientific notation with 2 decimal places
        cbar_kws={
            'label': 'Difference (Symmetric Log Scale)',
            'extend': 'both'  # Allow colorbars to extend beyond the main colors
        },
        
    )
    
    plt.title('bfloat16 L-Mul Error Heatmap (Symmetric Log Scale)')
    plt.xlabel('Input 1')
    plt.ylabel('Input 2')
    plt.xticks(rotation=45)
    plt.tight_layout()
    # plt.ticklabel_format(style='scientific', axis='both')
    
    return plt.gcf()

if __name__=="__main__":
    # Basic test cases setup
    special_bf16_test_cases = create_bf16_lmul_testbench(
        input_names=["fp_a", "fp_b"], 
        output_name="fp_out",
        n_values=0,
    )
    bf16_special_inputs, bf16_special_outputs = special_bf16_test_cases["inputs"], special_bf16_test_cases["outputs"]

    basic_bf16_test_cases = create_bf16_lmul_testbench(
        input_names=["fp_a", "fp_b"],
        output_name="fp_out",
        include_special_values=False,
    )
    bf16_basic_inputs, bf16_basic_outputs = basic_bf16_test_cases["inputs"], basic_bf16_test_cases["outputs"]

    results_df = run_testbench(
        rtl_func=bf16_lmul_naive, 
        input_names=["fp_a", "fp_b"],
        output_name="fp_out",
        start=-4,
        stop=0,
        n_values=25,
        include_special_values=False,
    )

    pivoted_df = reshape_testbench_results(results_df)
    fig = create_diff_heatmap_a(pivoted_df)
    # fig_alt = create_diff_heatmap_alt(pivoted_df)
    plt.savefig(OUTPUT_DIR)
    print("Saved ")