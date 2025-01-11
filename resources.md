
## Project Resources

1. Architectural level planning, requirements analysis, specification design, and division of work

   > - Google TPU ([paper](https://arxiv.org/abs/1704.04760), [blog](https://cloud.google.com/blog/products/ai-machine-learning/an-in-depth-look-at-googles-first-tensor-processing-unit-tpu))
   > - AWS Inf2 [architecture](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/arch/neuron-hardware/inferentia2.html)
   > - NVIDIA H100 [tensor core architecture](https://resources.nvidia.com/en-us-tensor-core)
   > - [Etched Transformer ASIC](https://www.etched.com/announcing-etched)

2. The hardware itself containing the logic for specialized modules/operations

- [PyRTL](https://sites.cs.ucsb.edu/~sherwood/pubs/FPL-17-pyrtl.pdf) is an easy to use hardware description language that allows us to generate and simulate hardware in Python and automatically generate Verilog code as output
- [Amaranth](https://amaranth-lang.org/docs/amaranth/) (previously nMigen) is similar to PyRTL but more mature, maintained, and provides better integration with tooling. I think it is worth considering switching to avoid potential limitations with PyRTL
  > - [example ALU in PyRTL](https://github.com/pllab/pipelined-alu/blob/master/README.md)
  > - [example RISCV CPU in PyRTL](https://github.com/pllab/BD-PyRTL-RV)

3. A set of opcodes/instructions that can interface with simulated memory

- [OWL](https://zsisco.net/papers/control-logic-synthesis.pdf) is a tool from the same lab that made PyRTL that automatically generates control logic for a given datapath ([github](https://github.com/UCSBarchlab/owl))
- [Instruction level abstraction](https://arxiv.org/pdf/1801.01114) allows us to generate PyRTL code with OWL by describing the behavior of the instruction set ([github]())
  > - [Example opcodes](https://github.com/pllab/embedded-class-riscv/blob/master/src/control.py) for a RISCV CPU in PyRTL (different project)

4. A fast, easy to use, and interactive simulation of the hardware that allows us to easily test sending inputs and getting outputs

- [Yosys](https://yosyshq.readthedocs.io/projects/yosys/en/latest/) is a framework for Verilog RTL synthesis.
- [Surfer VSCode extension](https://marketplace.visualstudio.com/items?itemName=surfer-project.surfer) lets you easily view the waveform outputs of simulation tests
- [OSS CAD Suite](https://github.com/YosysHQ/oss-cad-suite-build) is a binary software distribution for a number of open source software used in digital logic design. You will find tools for RTL synthesis, formal hardware verification, place & route, FPGA programming, and testing with support for HDLs like Verilog, Migen, and Amaranth.

5. An assembly language and assembler that compiles to the custom bytecode (Nvidia PTX)

   > - Ben Eater's [assembler](https://github.com/TheTask/8Bit-Assembler) for his [8-bit computer](https://eater.net/8bit) from scratch project

6. A custom programming language/library/SDK that compiles to our assembly lang (think CUDA, ROCm, Apple CoreML)

- [LLVM](https://llvm.org/docs/GettingStarted.html) is a compiler infrastructure that can be used to create a custom language, but this is probably overkill

7. Integrating the library with a machine learning framework or simulate using pure python.
   > We could integrate [ONNX](https://onnx.ai/onnx/intro/concepts.html), the open neural network exchange format which supports most modern models and hardware by extending the [onnxruntime](https://onnxruntime.ai/docs/reference/high-level-design.html) with a new [ExecutionProvider](https://onnxruntime.ai/docs/execution-providers/add-execution-provider.html) that uses our hardware accelerator. This would allow us to run almost any model in simulation. ONNX generates a computational graph of a model and the execution providers route subgraphs to accelerators based on the implemented kernels/operators.
