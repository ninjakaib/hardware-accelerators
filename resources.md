# Project Resources

## Todo List:
1. **Design base class for hardware simulations:**  
  All of the hardware modules should have a class that makes it easy to simulate them. The simulation classes should all share a common base class with some methods like simulate and build hardware. This will make testing them much easier. Something else we could do is create configuration dataclasses that make it easy to define a hardware spec (float dtype, lmul/regular multiplication, systolic array size, accumulator width, etc) and allows us to test a wide range of setups without rewriting a ton of parameters every time.

2. **Float type casting:**  
  Need to make a simple resuable hardware unit that can cast from one floating point type to another. Upcast is easy, downcast we need to choose a rounding method.

3. **Accumulator Buffer:**  
  Design another memory system that can be used for computing tiled matrix multiplications. This component will connect directly to the systolic array and accumulate the outputs at software specified addresses.

4. **Activation Module:**  
  Create a configurable activation function component that can easily be connected to other steps in the pipeline (matmul outputs) and supports the most commonly used functions like ReLU, Sigmoid, GeLU, Swish, Softmax, etc. This is essentially a vector/tensor processing unit.

5. **Memory Implementation:**  
  We need to design a memory system in PyRTL using `MemBlocks` that can store weights and activations to be used by other components like the systolic array. Since model weights will likely be too large to store completely in hardware memory (SRAM), we can emulate DRAM in software during simulation.

6. **ISA and Compiler Stack:**  
  We need to be able to take a ML model, and convert its inference steps into a sequence of operations supported by our hardware. We really need to solidify the overall chip architecture before defining an ISA. Once the ISA is done, we can create a compiler that turns models into instructions.

7. **Decoder and top level integration:**  
  We need to design a hardware decoder that takes binary instructions and maps them to various hardware components and operations. This is essentially the control unit that connects everything together (systolic array, activation module, accumulators, and memory)

8. **Analysis and Results:**  
  Time to finally simulate running models! Collect data for various configurations and combinations of different hardware units. Validate accuracy of operations at both individual component level and ML model level. Let's see if we can calculate the amount of compute/memory required to ensure our design optimizes the balance between computation and memory bandwidth limitations. We should also attempt to synthesize our design to estimate power, area, and delay statistics. It would be awesome if we could estimate some higher level stats like tokens/sec for an LLM.

9. **Create Website:**  
  We need to show off all the hard work we did in a way that's both meaningful/comprehensive and easy to understand. An interactive visualization of running a model and the active circuits in the synthesized hardware, and showing the results of the computation would be really cool. This is probably way too hard, so a demo that runs the simulation is probably good enough. We should also try to display a visualization of the chip design after generating a physical layout (GDSII).


---


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

---

## Bonus

1. Create a physical design after verifying simulations

- [OpenLane](https://openlane.readthedocs.io/en/latest/) is an automated RTL to GDSII flow based on several components including OpenROAD and Yosys
- [OpenROAD](https://github.com/The-OpenROAD-Project/OpenROAD) is the leading open-source, foundational application for semiconductor digital design. The OpenROAD flow delivers an Autonomous, No-Human-In-Loop (NHIL) flow, 24 hour turnaround from RTL-GDSII for rapid design exploration and physical design implementation.
- [IIC-OSIC-TOOLS](https://github.com/iic-jku/IIC-OSIC-TOOLS) is an all-in-one Docker image for SKY130/GF180/IHP130-based analog and digital chip design.

2. Include an interactive 3D model of the GDS files in the project

- [Tiny Tapeout](https://tinytapeout.com/) provides a [guide](https://tinytapeout.com/guides/workshop/create-your-gds/) for generating a github pages site with an interactive 3D model of the GDS files in the project
- [GDS2WebGL](https://github.com/s-holst/GDS2WebGL): This tool provides a performant, portable, and approachable way to visualize and browse chip layout data. It does so by translating the geometric shapes found in GDSII stream format into a self-contained HTML file that can be viewed in any modern WebGL-capable web browser.
- [Example](https://mattvenn.github.io/wokwi-verilog-gds-test/viewer/tinytapeout.html) GitHub pages project
