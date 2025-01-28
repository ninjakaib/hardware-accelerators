<div align="center">

# Designing an Energy Efficient Hardware Accelerator for Deep Learning

</div>

# Introduction
- problem description
- why is this important?
- what will the paper cover (outline)
- existing solutions (TPU, GPU, AWS inferentia, etc.)

---

# Review

## ML Models
- describe anatomy of models
- inference vs training
- review common operations, activation funcs, etc
### Transformers
- architecture overview
- mathematical operation analysis

## Quantization
- what it is and why its used
- integer quantization techniques
- floating point quantization techniques

## Floating Point Numbers
- internal representation
- different formats
- Subnormal numbers and special values
### Casting and conversion
- describe how floats are converted from one format to another

## Floating Point Arithmetic
### Addition
- describe addition algorithm
### Multiplication
- describe multiplication algorithm
### L-mul
- describe lmul algorithm

## Systolic Arrays
- describe matrix multiplication
- describe systolic array
### Data Flow Patterns
- weight stationary
- input stationary
- output stationary
- [DiP](https://arxiv.org/html/2412.09709v1)

## Hardware Accelerators
- existing solutions (TPU, GPU, AWS inferentia, etc.)
- describe architectures and programming/execution models
- benefits and limitations of each

---

# Methods

## Hardware Design
- discuss how accelerators are designed, simulated, tested, and verified
### PyRTL
- brief history overview, what it can do, who its made for
- how we used it in our project (design and simulation)

### Chip Architecture
- give high level overview of all modules
- describe computational dataflow
- ISA overview

### Low Level Design
- describe hardware implementation of each unit in detail
#### Adder
- pipeline vs simple
#### Multiplier
- pipeline vs simple
#### L-mul
- Handling special values (describe design choices and justifications)
#### Fused LmulAdd
- can include if we have time to implement

### Systolic Array
- describe various dataflow implementations
#### FIFO

### Accumulator Buffer

### Activation Modules
#### ReLU
#### Sigmoid
#### GeLU
#### Swish
#### SwiGLU

### Softmax

## Software Design

### Model parsing
- describe ONNX format, computational graphs
- describe compiler that maps model to chip

### Simulation
- how simulation works
- automatic testing for optimal configuration 

### Automated Synthesis
- describe how yosys, openlane, and other tools are used
- what is synthesis, place and route, etc

---

# Results
- individual hardware component analysis (lmul accuracy)
- systolic array performance tradeoffs and time complexities
- hardware stats (power, area, delay)
- performance (model accuracy, compute cycles)
- estimated results (LLM tokens/sec)

---

# Discussion
- interpretation of results
- what do they mean?
- compare to other works
- describe impact
- cover limitations

---

# Conclusion
