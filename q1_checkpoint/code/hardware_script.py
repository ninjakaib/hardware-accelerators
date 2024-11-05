import pyrtl
import os
from lmul_hardware import lmul_rtl
from IPython.display import display_svg

# Create output directory if it doesn't exist
os.makedirs('output', exist_ok=True)

# Create the hardware
pyrtl.reset_working_block()
fp_a, fp_b, fp_out = lmul_rtl()

# Save the SVG
svg = pyrtl.block_to_svg(maintain_arg_order=True)
with open('output/block_diagram.svg', 'w') as f:
    f.write(svg)

# Open the output file with UTF-8 encoding
with open('output/hardware_output.txt', 'w', encoding='utf-8') as f:
    # Redirect prints to file
    def write_to_file(text):
        f.write(text + '\n')

    pyrtl.synthesize()

    # Generating timing analysis information
    write_to_file("Pre Optimization:")
    timing = pyrtl.TimingAnalysis()
    original_max_length = timing.max_length()
    write_to_file(f"Critical path length: {original_max_length}")
    logic_area, mem_area = pyrtl.area_estimation(tech_in_nm=65)
    est_area = logic_area + mem_area
    write_to_file(f"Estimated Area of block {est_area} sq mm")
    write_to_file("")

    pyrtl.optimize()

    write_to_file("Post Optimization:")
    timing = pyrtl.TimingAnalysis()
    optimized_max_length = timing.max_length()
    write_to_file(f"Critical path length: {optimized_max_length}")
    logic_area, mem_area = pyrtl.area_estimation(tech_in_nm=65)
    est_area = logic_area + mem_area
    write_to_file(f"Estimated Area of block {est_area} sq mm")
    write_to_file("")

    # Set up simulation
    sim_trace = pyrtl.SimulationTrace()
    sim = pyrtl.Simulation(tracer=sim_trace)

    # Test vectors
    test_vectors = [
        # Normal case: two small numbers
        (0x40, 0x40),  # inputs have exponent=4, mantissa=0
        
        # Overflow case: two large numbers
        (0x70, 0x70),  # inputs have large exponents
        
        # Underflow case: two small numbers
        (0x01, 0x01),  # inputs have very small values
        
        # Mixed signs
        (0xC0, 0x40),  # negative * positive
    ]

    # Run simulation
    for a, b in test_vectors:
        sim.step({
            'fp_a': a,
            'fp_b': b
        })

    # Save trace to file
    write_to_file("\nSimulation Trace:")
    
    # Manually format the trace data
    num_steps = len(test_vectors)
    
    # Header row with step numbers
    header = "      "
    for i in range(num_steps):
        header += f"|{i:<5}"
    write_to_file(header)
    
    # Separator line
    separator = "------" + "------" * num_steps
    write_to_file(separator)
    
    # Data rows
    trace_data = sim_trace.trace
    for signal in ['fp_a', 'fp_b', 'fp_out']:
        row = f"{signal:>6} "
        for i in range(num_steps):
            value = trace_data[signal][i]
            row += f"|0x{value:02x} "
        write_to_file(row)

    # Print detailed results
    write_to_file("\nDetailed Results:")
    for i, (a, b) in enumerate(test_vectors):
        result = sim_trace.trace['fp_out'][i]
        write_to_file(f"fp_a: {format(a, '08b')} * fp_b: {format(b, '08b')} = fp_out: {format(result, '08b')}")