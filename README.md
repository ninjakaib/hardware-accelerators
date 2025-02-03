<div align="center">

# Hardware Accelerators


### UCSD DSC180B Capstone Project | Winter 2025

<!-- [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md) -->

</div>

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

## Team

**Mentor**: Professor Rajesh Gupta

**Contributors**:

- Kai Breese
- Justin Chou
- Katelyn Abille
- Lukas Fullner

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

This project follows the [GitHub Flow](https://docs.github.com/en/get-started/quickstart/github-flow) branching strategy. The `main` branch is locked to prevent direct pushes - all changes must be made through pull requests.

### Development Environment Setup

#### Dev Containers

To use the development environment, open VSCode and install the [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers). From there, the Dev Container options will be available in the bottom left of VSCode. Click on the icon in the status bar (it will say "Open a Remote Window"), or open the command palette with `Cmd/Ctrl+Shift+P`. Select "Reopen in container" to build and run the container. You should be all set to begin working on the project now!

#### Python Formatting

We use the Black formatter to maintain consistent code style. It is configured by default in the devcontainer. If you are working outside the container, you can configure VSCode to automatically format Python code on save by following these steps:

1. Install the Black Formatter extension from the VSCode marketplace
2. Open VSCode Settings (Command Palette → "Preferences: Open Settings (JSON)")
3. Add the following configuration:

```json
{
  "[python]": {
    "editor.defaultFormatter": "ms-python.black-formatter",
    "editor.formatOnSave": true
  }
}
```

This will automatically format Python files when you save them, ensuring consistent code style across the project.

> **Note**: Make sure you have Black installed in your development environment: `pip install black`

### Branch Guidelines

- Create feature branches for specific tasks/features
- Branch names should be descriptive and follow the pattern: `feature/` or `fix/` followed by what you're working on
- Keep branches focused on single features/fixes to maintain clean version control
- Commit code frequently with clear commit messages

Good branch names: `feature/systolic-memory-controller` or `fix/latex-workflow`  
Bad branch names: `kais-dev-branch` or `feature/misc-changes`

### Working with Branches

Create and switch to a new feature branch:

```bash
# Create and checkout new branch
git checkout -b feature/your-feature-name main

# Set upstream to track remote branch
git push -u origin feature/your-feature-name
```

Keep your feature branch up to date with main:

```bash
# Fetch latest changes
git fetch origin main

# Merge main into your feature branch
git merge origin/main

# Push updates to your feature branch
git push
```

> **Tip**: Update your feature branch from main frequently (daily or every few days) to catch conflicts early and keep them manageable.

### Pull Request Process

1. Ensure your code is well-tested and documented
2. Create a pull request against the `main` branch
3. PRs require 2 approvals from other contributors before merging
4. Use the PR description to explain your changes and any important considerations
5. Reviewers should provide constructive feedback and test the changes locally if needed
6. Before merging, ensure your branch is up to date with main

Remember to commit code frequently and keep your branches focused on specific tasks. This helps maintain a clear version history and makes code review easier for everyone.

After a pull request is merged into `main`, your branch will automatically be deleted. You can update your local repository to reflect these changes with `git fetch --prune`.
