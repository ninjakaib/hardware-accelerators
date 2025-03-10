
<div align="center">

# Hardware Accelerators

## UCSD DSC180B Capstone Project | Winter 2025

[![Website](https://img.shields.io/badge/View_Project_Website-blue?style=for-the-badge&logo=github)](https://nakschou.github.io/hardware-accelerators-site/)
[![Download Report](https://img.shields.io/badge/📄_Download_Report-red?style=for-the-badge&logo=adobe-acrobat-reader)](https://github.com/nakschou/artifact-directory-template/blob/main/report.pdf)

[![GitHub deployments](https://img.shields.io/github/deployments/ninjakaib/hardware-accelerators/github-pages?label=pages)](https://github.com/ninjakaib/hardware-accelerators/deployments/github-pages)
[![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/ninjakaib/hardware-accelerators/format-check.yml?label=formatting)](https://github.com/ninjakaib/hardware-accelerators/actions/workflows/format-check.yml)
[![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/ninjakaib/hardware-accelerators/pytest.yml?label=tests)](https://github.com/ninjakaib/hardware-accelerators/actions/workflows/pytest.yml)

</div>

## Team

**Mentor**: Professor Rajesh Gupta

**Contributors**:

- Kai Breese
- Justin Chou
- Katelyn Abille
- Lukas Fullner
---

## Q2 Final Overview

For our Q2 project, we built hardware simulations for both the IEEE and L-mul algorithms using PyRTL, which we analyzed using OpenROAD for our key efficiency and speed metrics (power, area, delay). It culminates in a gradio demo (also viewable on our site) where you can interact with the simulations, which are running an MLP for the MNIST dataset.

#### Project structure

- `./hardware_accelerators` contains all source code (PyRTL, NN, etc.) for our hardware implementation
- `./tests` will contain `pytest` tests that are automatically run as part of a CI pipeline
- `./notebooks` contains jupyter notebook files. The main branch should only have high quality, readable, reproduceable notebooks that can be used to help write the final report.
- `./openroad` contains the scripts for data transfer between an openroad container. 
- `./verilog` contains the Verilog for our hardware generated via PyRTL.
- `./models` contains the MNIST MLP for several data formats.

#### Reproducing our Demo
To reproduce our demo, you'll need Docker installed on your PC and have the daemon open. Then follow these steps:

1. Pull the docker image using: `docker pull nakschou/lmul-demo:latest`
2. Run the docker image using `docker pull nakschou/lmul-demo:latest`
3. Navigate to `localhost:8000` to view our demo
The output should be the result of our test cases.

#### Reproducing the OpenROAD Analysis
To reproduce our OpenROAD analysis, go ahead and build the docker container based on these instructions [here](https://openroad-flow-scripts.readthedocs.io/en/latest/user/DockerShell.html). From there, you can transfer the verilog already existing in this repository to the proper folder by specifying in [this file](https://github.com/ninjakaib/hardware-accelerators/blob/main/openroad/data_transfer/config.py) where your OpenROAD-flow-scripts/ is (the `WSL_OPENROAD_BASE`, and run the docker container from there.

#### Reproducing the Accuracy

To reproduce our accuracy figures, create a conda environment for Python 3.12 using: `conda env create -n yay python=3.12 -y`. Then, activate the environment using `conda activate yay`. From there, go to the root and run `pip install -r requirements.txt`. Then you can use our mnist_eval.py file to reproduce our accuracy scores.

---
## Q2 Checkpoint

For evaluation of our code, please look primarily at our `./hardware_accelerators` and `./tests` folders. Within those, we have some of our migrated code from Q1 in addition to some new features we've added -- namely, the systolic array, ONNX hardware simulation, and accumulators logic.

The PyRTL portion of our Quarter 2 project code had a few major implementations:

- An accumulator, which we initialize and simulate in our pytest
- The floating point adders, which we generated testcases for
- The BaseFloat abstract base class, which verifies that it works and cannot be instantiated
- The BF16 data class, which we test with special values, arithmetic operations, and edge cases
- The FP8 data class, which we test with special values, arithmetic operations, and edge cases
- The ONNX hardware simulator, which we test by loading in a model we've tested on and ensuring its outputs are correct

We've also implemented the following but have yet to build out the test suite for the following:

- The systolic array and accumulator buffer
- Systolic array tiling logic
- SwiGLU activation block

To reproduce our tests, you'll need Docker installed on your PC and have the daemon open. Then follow these steps:

1. Pull the docker image using: `docker pull nakschou/hardware_accelerators:q2_checkpoint`
2. Run the docker image using `docker run nakschou/hardware_accelerators:q2_checkpoint`

The output should be the result of our test cases.

---

## Project Overview



This project explores the implementation of the Linear-complexity Multiplication (L-Mul) algorithm for efficient floating-point multiplication on ASICs. As modern AI systems grow in scale and complexity, their computational processes require increasingly large amounts of energy. For example, ChatGPT required an estimated 564 MWh per day as of February 2023, with inference costs significantly outweighing training costs in the long term.

### The L-Mul Algorithm

The core idea of L-Mul is to eliminate the costly mantissa multiplication step in floating-point multiplication by approximating it with a simpler term L(M), where M is the number of mantissa bits. This approximation achieves high precision while significantly reducing computational overhead compared to traditional floating-point multiplication methods.

### Quarter 1 Accomplishments

In Q1, we:

- Implemented the L-Mul algorithm in PyRTL and Verilog
- Developed a 2x2 systolic array for matrix multiplication using L-Mul
- Validated the algorithm's accuracy against standard floating-point multiplication
- Created basic components for handling BFloat16 numbers
- Successfully demonstrated that theoretical simplifications translate to practical hardware designs

### Quarter 2 Work

Building on our Q1 work, we have:

- Expanded and optimized our systolic array implementation
- Built hardware activation units for activation functions commonly used in machine learning
- Benchmarked performance against traditional floating-point multiplication
- Run an MNIST MLP on simulated hardware

### Why This Matters

By optimizing the fundamental operation of floating-point multiplication, we can significantly reduce the energy consumption and processing time of neural network operations. This has important implications for making AI systems more environmentally sustainable and cost-effective, particularly in the inference phase where computational costs are highest.

For more details about our Q1 work, see our [technical report](reports/main.pdf).


---
## Hardware Simulation & Caching

The simulator uses just-in-time compilation with intelligent caching to accelerate hardware simulations. This system automatically:

1. Generates optimized C code for your hardware configuration
2. Compiles to native binaries for your platform
3. Caches compiled artifacts for future reuse

### Key Features

- **Automatic Cache Management**  
  Binaries are stored in platform-appropriate locations:
  - Linux: `~/.cache/hardware_accelerators`
  - macOS: `~/Library/Caches/HardwareAccelerators`
  - Windows: `%LOCALAPPDATA%\hardware_accelerators\Cache`
- **Environment Configuration**  
  Control cache location via:

  ```bash
  export HWA_CACHE_DIR=/path/to/custom/cache
  ```

  or using a `.env` file in your project root.

---
## Contributing

See more details in the [contributing](CONTRIBUTING.md) section.

## Extra Info

See some of the things that helped us, and other cool or similar projects on the [resources](resources.md) page.
