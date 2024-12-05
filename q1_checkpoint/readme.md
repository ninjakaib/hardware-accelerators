## Checkpoint

## Q1 Checkpoint
Our Q1 checkpoint code had two major components: a high-level implementation of the $Lmul$ algorithm introduced in the Addition is All You Need paper, and a basic hardware implementation using PyRTL. To reproduce our code, you need Docker installed on your PC. Then follow these steps: 

1. Pull the docker image using:
`docker pull nakschou/hardware_accelerators:q1_checkpoint`
2. Make an output directory using: `mkdir output`
3. Run the docker image using `docker run -v ${PWD}/output:/app/output nakschou/hardware_accelerators:q1_checkpoint` (assuming you are using Windows Powershell, otherwise use your proper current working directory.)

We also have our code and write-up in the `checkpoint.ipynb` file within the `q1_checkpoint/code` directory if you would like to follow our thought process. This has the same outputs as our docker image, but you must first `pip install -r requirements.txt` within that directory.

For the first portion, we had three output files (within the `q1_checkpoint/code/output` directory:
- `fp8_multiplication_error.png` -- a heatmap of the errors of our basic $Lmul$ implementation, using all FP8 numbers.
- `basic_lmul_normalxsubnormal.png` -- a heatmap of the errors of our basic $Lmul$ implementation when multiplying a normal and a subnormal.
- `optimized_lmul_normalxsubnormal.png` -- a heatmap of the rrors of our optimized $Lmul$ implementation when multiplying a normal and a subnormal.

The second portion had two output files:
- `block_diagram.svg` -- a block diagram of our basic hardware implementation
- `hardware_output.txt` -- all of our printed outputs from that script.

See organized code and report under [q1_checkpoint](/q1_checkpoint)