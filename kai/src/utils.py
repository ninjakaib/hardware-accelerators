import pyrtl
from IPython.display import display, Latex, display_svg
from typing import Annotated, Callable, Dict, List, Literal
import subprocess
import os
import sys
import datetime
from pyrtl.simulation import default_renderer

def _circuit_analysis(block=None, tech_in_nm=130):
    timing = pyrtl.TimingAnalysis(block=block)
    timing.print_max_length()
    max_freq = timing.max_freq(tech_in_nm=tech_in_nm)
    print("Max frequency of block: ", max_freq, "MHz")
    logic_area, mem_area = pyrtl.area_estimation(tech_in_nm=tech_in_nm, block=block)
    est_area = logic_area + mem_area
    print("Estimated Area of block", est_area, f"mm^2 using {tech_in_nm}nm process")
    print()

def basic_circuit_analysis(synthesize=True, optimize=True, tech_in_nm=130):
    print("Pre-synthesis Results:")
    _circuit_analysis(tech_in_nm=tech_in_nm)

    if synthesize:
        synth_block = pyrtl.synthesize()
        # Generating timing analysis information
        print("Synthesis Results:")
        _circuit_analysis(synth_block, tech_in_nm)
        
    if optimize:
        opt_block = pyrtl.optimize()
        _circuit_analysis(opt_block, tech_in_nm)


SaveTypes = Literal['svg', 'vcd', 'verilog']
SaveAs = Annotated[SaveTypes | list[SaveTypes], "Save format(s) for the circuit"]

def get_repo_root():
    try:
        return subprocess.check_output(['git', 'rev-parse', '--show-toplevel'], 
                                        universal_newlines=True).strip()
    except subprocess.CalledProcessError:
        return None
    
def custom_render_trace(
    trace, 
    trace_list: List[str] = None, 
    file=sys.stdout,
    renderer = default_renderer(),
    symbol_len: int = None,
    repr_func: Callable[[int], str] = hex,
    repr_per_name = {},
    segment_size: int = 1
):
    from IPython.display import display, HTML, Javascript  # pylint: disable=import-error
    htmlstring = pyrtl.trace_to_html(trace, trace_list=trace_list, repr_func=repr_func, repr_per_name=repr_per_name)
    html_elem = HTML(htmlstring)
    display(html_elem)
    js_stuff = """
    $.when(
    $.getScript("https://cdnjs.cloudflare.com/ajax/libs/wavedrom/1.6.2/skins/default.js"),
    $.getScript("https://cdnjs.cloudflare.com/ajax/libs/wavedrom/1.6.2/wavedrom.min.js"),
    $.Deferred(function( deferred ){
        $( deferred.resolve );
    })).done(function(){
        WaveDrom.ProcessAll();
    });"""
    display(Javascript(js_stuff))

def analyze_circuit(
        circuit_func: callable, 
        svg: bool = False, 
        split_state: bool = False,
        display_pre_opt: bool = False,
        sim: bool = False,
        inputs: dict[str, list[int]] | None = None, 
        outputs: list[str] | dict[str, list[int]] | None = None,
        trace_list: list[str] = [],
        repr_per_name = {},
        save: SaveTypes | list[SaveTypes] = [],
        path_to_output_dir: str = './output',
    ):    
    # Create the hardware
    pyrtl.reset_working_block()
    rtl = circuit_func()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if isinstance(save, str):
        save = [save]

    if svg and display_pre_opt:
        svg = pyrtl.block_to_svg(split_state=split_state)
        display_svg(svg, raw=True)

    pyrtl.synthesize()

    # Generating timing analysis information
    print("Pre Optimization:")
    timing = pyrtl.TimingAnalysis()
    timing.print_max_length()
    logic_area, mem_area = pyrtl.area_estimation(tech_in_nm=65)
    est_area = logic_area + mem_area
    print("Estimated Area of block", est_area, "sq mm")
    print()

    pyrtl.optimize()

    print("Post Optimization:")
    timing = pyrtl.TimingAnalysis()
    timing.print_max_length()
    logic_area, mem_area = pyrtl.area_estimation(tech_in_nm=65)
    est_area = logic_area + mem_area
    print("Estimated Area of block", est_area, "sq mm")
    print()

    # Print the max clk frequency
    max_freq = timing.max_freq()
    print("Max frequency of block: ", max_freq, "MHz")
    print()

    if svg and not display_pre_opt:
        svg = pyrtl.block_to_svg(split_state=split_state)
        display_svg(svg, raw=True)
    
    # Set up simulation
    if sim:
        sim_trace = pyrtl.SimulationTrace()
        sim = pyrtl.Simulation(tracer=sim_trace)

        if inputs is not None:
            # Set up wires to trace in the simulation
            trace_list = list(inputs.keys()) + trace_list
            if isinstance(outputs, list):
                trace_list.extend(outputs)
                outputs = {}
            elif isinstance(outputs, dict):
                trace_list.extend(list(outputs.keys()))
            # Remove potential duplicates
            trace_list = [t for i, t in enumerate(trace_list) if t not in trace_list[:i]]

            sim.step_multiple(inputs, outputs)

            sim_trace.render_trace(
                trace_list=trace_list,
                repr_per_name=repr_per_name
            )

            if 'vcd' in save:
                trace_output_path = os.path.join(path_to_output_dir, 'simulations', f'{circuit_func.__name__}_{timestamp}.vcd')
                with open(trace_output_path, 'w') as f:
                    sim_trace.print_vcd(f, include_clock=True)
                    print(f"Simulation trace saved to {trace_output_path}")

    if svg and 'svg' in save:
        svg_output_path = os.path.join(path_to_output_dir, 'assets', f'{circuit_func.__name__}_{timestamp}.svg')
        with open(svg_output_path, 'w') as f:
            f.write(svg)

    if 'verilog' in save:
        verilog_output_path = os.path.join(path_to_output_dir, 'verilog', f'{circuit_func.__name__}_{timestamp}.v')
        with open(verilog_output_path, 'w') as f:
            pyrtl.output_to_verilog(f)



def display_sign_steps(bits):
    sign_bit = bits[0]
    display(Latex(f"Sign Bit: ${sign_bit}$"))
    display(Latex(f"$(-1)^{sign_bit} = {'-' if sign_bit=='1' else '+'}1$"))

def display_exponent_steps(bits, format="E4M3"):
    if format == "E4M3":
        exp_bits = bits[1:5]
        bias = 7
    else: # E5M2
        exp_bits = bits[1:6] 
        bias = 15
        
    exp_val = int(exp_bits, 2)
    display(Latex(f"Exponent Bits: ${exp_bits}$"))
    display(Latex(f"$={exp_bits}_2 = {exp_val}_{{10}}$"))
    
    if exp_val == 0:
        # Subnormal case
        display(Latex(f"Subnormal case: $E = 0$ so use $2^{{1-bias}}$"))
        display(Latex(f"$2^{{1-{bias}}} = 2^{{{1-bias}}}$"))
    else:
        # Normal case
        display(Latex(f"$2^{{E-bias}} = 2^{{{exp_val}-{bias}}} = 2^{{{exp_val-bias}}}$"))

def display_mantissa_steps(bits, format="E4M3"):
    if format == "E4M3":
        mantissa_bits = bits[5:]
        m = 3
    else: # E5M2
        mantissa_bits = bits[6:]
        m = 2
        
    exp_val = int(bits[1:5], 2) if format == "E4M3" else int(bits[1:6], 2)
    
    display(Latex(f"Mantissa Bits: ${mantissa_bits}$"))
    
    M = int(mantissa_bits, 2)
    display(Latex(f"$M = {mantissa_bits}_2 = {M}_{{10}}$"))
    
    if exp_val == 0:
        # Subnormal case
        display(Latex(f"Subnormal case: $(0 + 2^{{-{m}}} × {M})$"))
        mantissa_val = (0 + 2**(-m) * M)
        display(Latex(f"$= {mantissa_val}$"))
    else:
        # Normal case
        display(Latex(f"$(1 + 2^{{-{m}}} × {M})$"))
        mantissa_val = (1 + 2**(-m) * M)
        display(Latex(f"$= {mantissa_val}$"))

def splice_bits(bits: str, format="E4M3"):
    if format == "E4M3":
        return bits[0] + " " + bits[1:5] + " " + bits[5:]
    else: # E5M2
        return  bits[0] + " " + bits[1:6] + " " + bits[6:]

def display_float8_conversion(bits, format="E4M3", verbose=False):
    if verbose:
        display(Latex(f"Converting {format} number: {splice_bits(bits, format)}"))
        display(Latex("Step 1: Sign"))
        display_sign_steps(bits)
        
        display(Latex("Step 2: Exponent"))
        display_exponent_steps(bits, format)
        
        display(Latex("Step 3: Mantissa"))
        display_mantissa_steps(bits, format)
    
    # Calculate final value
    sign_bit = bits[0]
    sign = -1 if sign_bit == '1' else 1
    
    if format == "E4M3":
        exp_bits = bits[1:5]
        mantissa_bits = bits[5:]
        bias = 7
        m = 3
    else:
        exp_bits = bits[1:6]
        mantissa_bits = bits[6:]
        bias = 15
        m = 2
        
    exp_val = int(exp_bits, 2)
    M = int(mantissa_bits, 2)
    
    if exp_val == 0 and M > 0:
        # Subnormal
        value = sign * (2**(1-bias)) * (0 + 2**(-m) * M)
    elif exp_val == 0 and M == 0:
        # Zero or negative zero
        value = 0
    else:
        # Normal
        value = sign * (2**(exp_val-bias)) * (1 + 2**(-m) * M)
        
    display(Latex(f"Final Value: $\\fbox{{{value}}}$"))
    return value


# display_float8_conversion('00000011')