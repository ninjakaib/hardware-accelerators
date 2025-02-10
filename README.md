<div align="center">

# Hardware Accelerators

## UCSD DSC180B Capstone Project | Winter 2025

[![Website](https://img.shields.io/badge/View_Project_Website-blue?style=for-the-badge&logo=github)](https://ninjakaib.github.io/hardware-accelerators/)
[![Download Report](https://img.shields.io/badge/📄_Download_Report-red?style=for-the-badge&logo=adobe-acrobat-reader)](https://github.com/ninjakaib/hardware-accelerators/raw/main/reports/main.pdf)

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

> [!NOTE]
> Website code is under the [`pages`](https://github.com/ninjakaib/hardware-accelerators/tree/pages) branch!

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

### Quarter 2 Goals

Building on our Q1 work, we aim to:

- Focus exclusively on PyRTL implementation for faster development
- Expand and optimize our systolic array implementation
- Build hardware activation units for activation functions commonly used in machine learning
- Benchmark performance against traditional floating-point multiplication
- Run models on simulated hardware

### Why This Matters

By optimizing the fundamental operation of floating-point multiplication, we can significantly reduce the energy consumption and processing time of neural network operations. This has important implications for making AI systems more environmentally sustainable and cost-effective, particularly in the inference phase where computational costs are highest.

For more details about our Q1 work, see our [technical report](reports/main.pdf).

---

## Project structure

- `./hardware_accelerators` will contain all source code (PyRTL, Verilog, etc.)
- `./tests` will contain `pytest` tests that are automatically run as part of a CI pipeline
- `./reports` contains the source LaTeX files for the report and the pdf generated by the github action
- `./notebooks` should hold all jupyter notebook files. The main branch should only have high quality, readable, reproduceable notebooks that can be used to help write the final report.

---

## Contributing

See more details in the [contributing](CONTRIBUTING.md) section.

## Extra Info

See some of the things that helped us, and other cool or similar projects on the [resources](resources.md) page.
