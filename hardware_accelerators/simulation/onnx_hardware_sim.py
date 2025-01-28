import os
from collections import defaultdict
from io import BytesIO
from typing import IO, Any, Callable, Dict, List

import numpy as np
import onnx
import pyrtl
import requests


class ONNXHardwareSimulator:
    def __init__(
        self,
        model: IO[bytes] | str | os.PathLike,
        operator_mapping: Dict[str, Callable],
    ):
        # Load ONNX model
        self.model = onnx.load(model)
        self.graph = self.model.graph

        # Store operator mapping
        self.op_mapping = operator_mapping

        # Initialize PyRTL
        pyrtl.reset_working_block()

        # Create input wire vectors
        # For MNIST: 28x28 = 784 inputs, assuming 8-bit precision
        self.input_wires = pyrtl.Input(bitwidth=8, name="input")

        # Create output wire vectors
        self.output_wires = pyrtl.Output(bitwidth=8, name="output")

        # Create memory blocks for weights
        self.weight_memories = {}
        self.setup_memories()

        # Track node execution state
        self.node_outputs = {}  # Store intermediate outputs
        self.current_node_idx = 0
        self.nodes = list(self.graph.node)

        # Build dependency graph for topological ordering
        self.dependencies = self._build_dependency_graph()
        self.execution_order = self._determine_execution_order()

        # Initialize PyRTL simulation
        self.sim = None

    def setup_memories(self):
        """Initialize memory blocks for weights and biases"""
        for init in self.graph.initializer:
            np_array = onnx.numpy_helper.to_array(init)
            # Create memory block based on initializer shape and type
            # For testing, we'll use 8-bit width for all memories
            mem = pyrtl.MemBlock(
                bitwidth=8, addrwidth=len(np_array.flatten()), name=init.name
            )
            self.weight_memories[init.name] = mem

    def _build_dependency_graph(self) -> Dict[str, List[str]]:
        """Build graph of node dependencies based on inputs/outputs"""
        deps = defaultdict(list)
        # Map output names to producing nodes
        output_to_node = {}

        for node in self.nodes:
            # Register this node as producer of its outputs
            for output in node.output:
                output_to_node[output] = node.name

            # Record dependencies based on inputs
            for input_name in node.input:
                if input_name in output_to_node:
                    deps[node.name].append(output_to_node[input_name])

        return deps

    def _determine_execution_order(self) -> List[int]:
        """Determine topological ordering of nodes for execution"""
        visited = set()
        order = []

        def visit(node_idx):
            if node_idx in visited:
                return
            visited.add(node_idx)

            node = self.nodes[node_idx]
            for dep in self.dependencies[node.name]:
                # Find index of dependent node
                dep_idx = next(i for i, n in enumerate(self.nodes) if n.name == dep)
                visit(dep_idx)
            order.append(node_idx)

        for i in range(len(self.nodes)):
            visit(i)
        return order

    def step(self) -> bool:
        """Execute one node in the computation graph
        Returns:
            bool: True if there are more nodes to process, False if done
        """
        if self.current_node_idx >= len(self.execution_order):
            return False

        # Get next node to execute
        node_idx = self.execution_order[self.current_node_idx]
        node = self.nodes[node_idx]

        # Get operator implementation
        op_impl = self.op_mapping[node.op_type]

        # Collect input values
        input_values = []
        for input_name in node.input:
            if input_name in self.node_outputs:
                # Get intermediate value from previous computation
                input_values.append(self.node_outputs[input_name])
            else:
                # Check if it's an initializer (weight/bias)
                init = self._get_initializer(input_name)
                if init is not None:
                    input_values.append(init)
                else:
                    # Must be network input
                    input_values.append(self.input_wires)

        # Execute operator
        outputs = op_impl(*input_values)

        # Store results
        if isinstance(outputs, tuple):
            for output_name, output_val in zip(node.output, outputs):
                self.node_outputs[output_name] = output_val
        else:
            self.node_outputs[node.output[0]] = outputs

        self.current_node_idx += 1
        return True

    def _get_initializer(self, name: str) -> pyrtl.WireVector | None:
        """Get initializer (weight/bias) value if it exists"""
        if name in self.weight_memories:
            mem = self.weight_memories[name]
            # Create an address wire to read from memory
            addr = pyrtl.Input(mem.addrwidth, name=f"{name}_addr")
            # Read from memory using the address
            return mem[addr]  # This returns a WireVector
        return None

    def reset(self):
        """Reset simulation state"""
        self.current_node_idx = 0
        self.node_outputs.clear()
        if self.sim:
            self.sim = pyrtl.Simulation()

    def __iter__(self):
        self.reset()
        return self

    def __next__(self):
        if not self.step():
            raise StopIteration
        return self.node_outputs


def mock_conv(x, weights, bias=None):
    """Mock convolution that just returns a wire vector of expected shape"""
    # Create a wire vector for output
    out = pyrtl.WireVector(bitwidth=8, name="conv_out")
    # Connect the output to prevent optimization removing it
    out <<= x  # For testing, just pass through input
    return out


def mock_maxpool(x):
    """Mock maxpool that just returns a wire vector of expected shape"""
    out = pyrtl.WireVector(bitwidth=8, name="pool_out")
    out <<= x  # For testing, just pass through input
    return out


def mock_relu(x):
    """Mock ReLU that just passes through the input"""
    out = pyrtl.WireVector(bitwidth=8, name="relu_out")
    out <<= x
    return out


def mock_matmul(x, weights):
    """Mock matrix multiplication that returns expected shape"""
    out = pyrtl.WireVector(bitwidth=8, name="matmul_out")
    out <<= x  # For testing, just pass through input
    return out


def mock_reshape(x, shape):
    """Mock reshape that just passes through the input"""
    out = pyrtl.WireVector(bitwidth=8, name="reshape_out")
    out <<= x
    return out


def mock_add(x, y):
    """Mock add operation that handles WireVectors properly"""
    if isinstance(x, pyrtl.MemBlock):
        x = x[pyrtl.Input(x.addrwidth, name="x_addr")]
    if isinstance(y, pyrtl.MemBlock):
        y = y[pyrtl.Input(y.addrwidth, name="y_addr")]
    return x + y


def test_simulator():
    # Create mock operator mapping with updated mock_add
    op_mapping = {
        "Conv": mock_conv,
        "MaxPool": mock_maxpool,
        "Relu": mock_relu,
        "Add": mock_add,  # Use our new mock_add instead of lambda
        "MatMul": mock_matmul,
        "Reshape": mock_reshape,
    }

    # Initialize simulator with MNIST model
    model = "model_data/mnist-12.onnx"
    if not os.path.exists(model):
        print(f"{model} not found locally, attempting download...")
        model = BytesIO(
            requests.get(
                "https://github.com/onnx/models/raw/refs/heads/main/validated/vision/classification/mnist/model/mnist-12.onnx?download="
            ).content
        )
    sim = ONNXHardwareSimulator(model, op_mapping)

    # Test iteration
    print("Testing step-by-step execution:")
    for i, step_results in enumerate(sim):
        print(f"Step {i}: Executed node {sim.nodes[sim.execution_order[i]].op_type}")

        # Print intermediate outputs
        print("  Outputs:", list(step_results.keys()))
        print()

    # Test reset and manual stepping
    print("\nTesting manual stepping:")
    sim.reset()
    step_count = 0
    while sim.step():
        node = sim.nodes[sim.execution_order[step_count]]
        print(f"Manual step {step_count}: {node.op_type}")
        step_count += 1

    # Verify execution order makes sense
    print("\nVerifying execution order:")
    execution_sequence = [sim.nodes[idx].op_type for idx in sim.execution_order]
    print("Operation sequence:", execution_sequence)

    # Test dependency graph
    print("\nDependency graph:")
    for node_name, deps in sim.dependencies.items():
        print(f"{node_name} depends on: {deps}")


if __name__ == "__main__":
    print(os.getcwd())
    test_simulator()
