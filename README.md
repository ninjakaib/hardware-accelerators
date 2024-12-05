
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