
# hardware-accelerators
UCSD DSC180 capstone project course work.

## Q1 Project 

The PyRTL portion of our Quarter 1 project code had three major elements: an implementation of an IEEE-754 multiplication pipeline, an implementation of a pipeline for the $Lmul$ algorithm introduced in the Addition is All You Need paper, and an implementation of an adder pipeline. All of the above aspects were created for the BF16 datatype. To reproduce our code, you need Docker installed on your PC. Then follow these steps: 

1. Pull the docker image using:
`docker pull nakschou/hardware_accelerators:q1_project`
2. Make an output directory using: `mkdir output`
3. Run the docker image using `docker run -v ${PWD}/output:/app/output nakschou/hardware_accelerators:q1_project` (assuming you are using Windows Powershell, otherwise use your proper current working directory.)

For the first portion, we had two output files (within the `q1_checkpoint/src/output` directory):
- `ieee_trace_output.html` -- a waveform trace of running several multiplication test cases through our algorithm.
- `ieee_visualization.svg` -- a visualization of our PyRTL pipeline in SVG format.

The second portion had x output files:
- `x.svg` -- lorem ipsum
- `x.txt` -- lorem ipsum

The second portion had y output files:
- `yay.svg` -- lorem ipsum
- `yay.txt` -- lorem ipsum

See organized code and report under [q1_project](/q1_project)

### Verilog Code
Alongside the PyRTL code we also created a systolic array using verilog.  The code would be difficult to run, involving installing the 260 GB Vivado program as well as setting up the project.  All of the verilog code is in the [IPGenerator src](/lukas/ipGeneratorProject.srcs/), with sims holding the test benches and sources holding the actual files.  We have rendered out the circuit diagrams for various modules located in the [Schematics](/lukas/Schematics/) folder.  The schematics are as follows:
- **Systolic**: this module is the systolic array.  It takes in two 2x2 matrices of bf16 floats, as well as a clock and reset signal.  When clocked three times it feeds the data from the matrices through the systolic array, performing a multiply and accumulate in each processing element.  The outputs are retrieved from the processing elements directly (the four grey blocks at the right of the diagram).  Finally a reset signal will clear all the internal buffers.
- **PE**: this module is the processing element.  It takes in two bf16 floats, then when it recieves a clock signal it multiplies them using lmul and then adds them to a stored accumulated value.  The outputs are stored in registers to be retireved by the systolic module, and it also stores the input floats so that it can pass them to the next elements in the systolic array.  Finally it also has a reset signal that will clear the accumulated value.
- **adder**: this is the bf16 adder.  This takes in two bf16 floats and directly performs a floating point addition.
- **lmul**: this is the lmul module.  It takes in two bf16 floats and multiplies them using the lmul algorithm.
