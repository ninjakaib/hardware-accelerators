import os
import pickle
import ctypes
import _ctypes
import platform
import shutil
from typing import Literal, Optional, Dict, List, Union, Any

import numpy as np
from pyrtl import (
    CompiledSimulation,
    SimulationTrace,
    PyrtlError,
    working_block,
    Block,
    Input,
    Output,
)

from ..nn.util import softmax

from ..nn.mlp import MLP

from .matrix_utils import (
    bias_trick,
    count_batch_gemm_tiles,
    generate_batch_gemm_tiles,
    generate_gemv_tiles,
    permutate_weight_matrix,
)

from ..rtllib.accelerator import (
    AcceleratorConfig,
    CompiledAccelerator,
    CompiledAcceleratorConfig,
)


class ReusableCompiledSimulation(CompiledSimulation):
    """Extension of CompiledSimulation that supports loading from precompiled libraries."""

    def __init__(self, lib_path: Optional[str] = None):
        """Initialize either from scratch or from a precompiled library.

        Args:
            lib_path: Path to precompiled library. If provided, loads from binary.
                     If None, initializes normally (compiles a new binary).
        """
        if lib_path is None:
            # Normal initialization - will compile the binary
            super().__init__()  # type: ignore
        else:
            # Load from precompiled library - skip compilation
            self._load_from_binary(lib_path)

    def _load_from_binary(self, lib_path: str):
        """Load simulation from a precompiled binary and state file.

        Args:
            lib_path: Path to either the .so file or the directory containing it.
        """
        # Check if lib_path is a directory or a file
        if os.path.isdir(lib_path):
            # It's a directory, look for pyrtlsim.so inside it
            so_path = os.path.join(lib_path, "pyrtlsim.so")
            state_path = os.path.join(lib_path, "state.pkl")
        else:
            # It's a file path, assume it's the .so file
            so_path = lib_path
            # Check if state file is in the same directory with generic name
            state_path = os.path.join(os.path.dirname(lib_path), "state.pkl")
            if not os.path.exists(state_path):
                # Try the old naming convention
                state_path = os.path.splitext(lib_path)[0] + ".state"

        # Verify files exist
        if not os.path.exists(so_path):
            raise PyrtlError(f"Library file not found: {so_path}")
        if not os.path.exists(state_path):
            raise PyrtlError(f"State file not found: {state_path}")

        # Load the saved state
        with open(state_path, "rb") as f:
            state = pickle.load(f)

        # Restore state
        self._dll = None
        self._dir = None
        self.block = state["block"]
        self.tracer = state["tracer"]
        self.default_value = state["default_value"]
        self._regmap = state["regmap"]
        self._memmap = state["memmap"]
        self._uid_counter = state["uid_counter"]
        self.varname = state["varname"]
        self._inputpos = state["inputpos"]
        self._outputpos = state["outputpos"]
        self._inputbw = state["inputbw"]
        self._ibufsz = state["ibufsz"]
        self._obufsz = state["obufsz"]
        self._probe_mapping = state.get("probe_mapping", {})

        # Load the DLL
        self._dll = ctypes.CDLL(lib_path)
        self._crun = self._dll.sim_run_all
        self._crun.restype = None
        self._initialize_mems = self._dll.initialize_mems
        self._initialize_mems.restype = None
        self._mem_lookup = self._dll.lookup
        self._mem_lookup.restype = ctypes.POINTER(ctypes.c_uint64)

        # Initialize memories
        self._initialize_mems()

    def save_compiled_lib(self, directory=None, name="default"):
        """Save the compiled library and state to a permanent location.

        Args:
            directory: Base directory to save the files to. If None, uses 'simulations/'.
            name: Name of the subfolder to store the files (default: 'default')

        Returns:
            Path to the directory where files were saved
        """
        if directory is None:
            directory = "simulations"

        # Create the specific directory for this simulation
        save_dir = os.path.join(directory, name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Ensure the DLL has been created
        if self._dll is None or self._dir is None:
            raise PyrtlError("No compiled library exists to save")

        # File paths with generic names
        lib_path = os.path.join(save_dir, "pyrtlsim.so")
        state_path = os.path.join(save_dir, "state.pkl")

        # Copy the library
        shutil.copy2(os.path.join(self._dir, "pyrtlsim.so"), lib_path)

        # Save state
        state = {
            "block": self.block,
            "tracer": self.tracer,
            "default_value": self.default_value,
            "regmap": self._regmap,
            "memmap": self._memmap,
            "uid_counter": self._uid_counter,
            "varname": self.varname,
            "inputpos": self._inputpos,
            "outputpos": self._outputpos,
            "inputbw": self._inputbw,
            "ibufsz": self._ibufsz,
            "obufsz": self._obufsz,
            "probe_mapping": getattr(self, "_probe_mapping", {}),
        }

        with open(state_path, "wb") as f:
            pickle.dump(state, f)

        return save_dir


class CompiledAcceleratorSimulator:
    """Simulator for the accelerator that uses compiled simulation for speed."""

    # Define a standard location for storing binaries
    BINARY_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "lib")

    def __init__(self, config_or_path: CompiledAcceleratorConfig | str):
        """Initialize the simulator either with a config or from a saved binary.

        Args:
            config_or_path: Either an AcceleratorConfig object or a path to a saved binary
        """
        # Create binary directory if it doesn't exist
        os.makedirs(self.BINARY_DIR, exist_ok=True)

        self.output_trace = []

        if isinstance(config_or_path, str):
            # Load from saved binary
            self._load_from_binary(config_or_path)
        else:
            # Initialize with config
            self.config = config_or_path
            self._setup_with_config()

    def _get_binary_path(self, config: CompiledAcceleratorConfig):
        """Get the path where binaries for this config should be stored."""
        # Create a unique identifier based on config parameters
        return os.path.join(self.BINARY_DIR, config.name)

    def _get_config_path(self, sim_dir):
        """Get the path where the config for this binary should be stored."""
        return os.path.join(self.BINARY_DIR, sim_dir, "config.pkl")

    def _load_from_binary(self, sim_dir):
        """Load simulator from a saved binary directory."""
        # Load the config
        config_path = self._get_config_path(sim_dir)
        if os.path.exists(config_path):
            print(f"Loading existing config from {config_path}")
            with open(config_path, "rb") as f:
                self.config = pickle.load(f)
        else:
            raise PyrtlError(f"Config file not found: {config_path}")

        # Create the simulation from the binary
        lib_path = os.path.join(self.BINARY_DIR, sim_dir, "pyrtlsim.so")
        self.sim = ReusableCompiledSimulation(lib_path=lib_path)

        # Extract output wires from the simulation's block
        # This assumes output wires follow the naming convention 'out_X'
        self.output_wires = []
        for wire in self.sim.block.wirevector_subset(Output):
            if wire.name.startswith("out_"):
                self.output_wires.append(wire)

    def _check_matching_binary(self):
        """Check if a binary for the current config already exists. If the folder exists and the config matches the current config, return True. Otherwise, return False."""
        pass

    def _setup_with_config(self):
        """Set up the hardware and simulation with the provided config."""
        # Check if we have a cached compiled library
        sim_dir = self._get_binary_path(self.config)
        lib_path = os.path.join(sim_dir, "pyrtlsim.so")

        if os.path.exists(lib_path):
            # Use the precompiled library
            print(f"Using precompiled library: {lib_path}")
            self.sim = ReusableCompiledSimulation(lib_path=lib_path)
        else:
            # Create and save a new compiled simulation
            self.construct_hardware()
            self.sim = ReusableCompiledSimulation()

            # Save the binary and state
            save_dir = self.sim.save_compiled_lib(
                directory=self.BINARY_DIR, name=os.path.basename(sim_dir)
            )

            # Save the config
            with open(self._get_config_path(save_dir), "wb") as f:
                pickle.dump(self.config, f)

    def construct_hardware(self):
        """Construct the hardware for the accelerator."""
        print(f"Constructing hardware for config {self.config.name}...")
        self.accelerator = CompiledAccelerator(self.config)
        # Create input and output wires
        inputs = {
            "data_enable": Input(1, "data_enable"),
            "data_inputs": [
                Input(self.config.activation_type.bitwidth(), f"data_in_{i}")
                for i in range(self.config.array_size)
            ],
            "weight_enable": Input(1, "weight_enable"),
            "weights_in": [
                Input(self.config.weight_type.bitwidth(), f"weight_in_{i}")
                for i in range(self.config.array_size)
            ],
            "accum_addr": Input(self.config.accum_addr_width, "accum_addr"),
            "accum_mode": Input(1, "accum_mode"),
            "act_start": Input(1, "act_start"),
            "act_func": Input(1, "act_func"),
        }
        self.accelerator.connect_inputs(**inputs)
        self.output_wires = [
            Output(self.config.activation_type.bitwidth(), f"out_{i}")
            for i in range(self.config.array_size)
        ]
        self.accelerator.connect_outputs(self.output_wires, Output(1, "output_valid"))

    def reset_sim(self):
        """Reset the simulation"""
        self.reset_output_trace()

        # Recreate the simulation
        sim_dir = self._get_binary_path(self.config)
        lib_path = os.path.join(sim_dir, "pyrtlsim.so")
        if os.path.exists(lib_path):
            # Use the precompiled library
            self.sim = ReusableCompiledSimulation(lib_path=lib_path)
        else:
            # Create a new simulation
            self.sim = ReusableCompiledSimulation()

    def get_sim_inputs(self, **kwargs) -> Dict[str, int]:
        """Get default input values"""
        defaults = {
            "data_enable": 0,
            "weight_enable": 0,
            "accum_addr": 0,
            "accum_mode": 0,
            "act_start": 0,
            "act_func": 0,
            **{f"data_in_{i}": 0 for i in range(self.config.array_size)},
            **{f"weight_in_{i}": 0 for i in range(self.config.array_size)},
        }
        for key, value in kwargs.items():
            if key in defaults:
                defaults[key] = value
            else:
                Warning(f"Invalid input key: {key}")
        return defaults

    def step(
        self,
        **kwargs,
    ) -> None:
        """Execute a simulation step with the given inputs.

        Args:
            data: Input data vector of length array_size. If provided, data_enable will be set to 1
            **kwargs: Additional control signals
        """
        data_vec = kwargs.pop("data_vec", None)
        weight_vec = kwargs.pop("weight_vec", None)
        inputs = self.get_sim_inputs(**kwargs)

        # Handle input vectors
        if weight_vec is not None:
            if len(weight_vec) != self.config.array_size:
                raise ValueError(
                    f"Weight vector length must be {self.config.array_size}"
                )
            inputs["weight_enable"] = 1
            for i, value in enumerate(weight_vec):
                inputs[f"weight_in_{i}"] = self.config.weight_type(value).binint
        if data_vec is not None:
            if len(data_vec) != self.config.array_size:
                raise ValueError(f"Data vector length must be {self.config.array_size}")
            inputs["data_enable"] = 1
            for i, value in enumerate(data_vec):
                inputs[f"data_in_{i}"] = self.config.activation_type(value).binint

        self.sim.step(inputs)
        if self.sim.inspect("output_valid") == 1:
            self.output_trace.append(self.inspect_outputs())

    def execute_instruction(
        self,
        weights: np.ndarray,
        data: np.ndarray,
        accum_addr: int = 0,
        accum_mode: int = 0,  # 0=overwrite, 1=accumulate
        activation_enable: bool = False,  # 0=nothing, 1=enable
        activation_func: Optional[Literal["relu"]] = None,
        flush_pipeline: bool = True,
    ) -> None:
        """Execute one instruction with the given inputs. Will step more than one cycle

        Args:
            data: Input data vector of length array_size. If None, data_enable will be 0
            accum_addr: Address in accumulator memory to start writing to. Auto increments with vector length
            accum_mode: 0 for overwrite, 1 for accumulate with existing values
            activation_enable: Whether to enable the activation unit
            activation_func: Activation function to use (passthrough or ReLU)
            load_new_weights: Whether to load a new weight tile from FIFO memory
            weight_tile_addr: Address of weight tile to load when weights=1
            flush_pipeline: Whether to flush the systolic array after input data to allow for new data/weights on the next cycle. If False, the systolic array will be held in the same state for the next cycle allowing for multiple data inputs to be streamed in.
            nop: No operation, step 1 cycle with no inputs
        """

        assert weights.shape == (
            self.config.array_size,
            self.config.array_size,
        ), f"Weights must be {self.config.array_size}x{self.config.array_size}"
        weights = permutate_weight_matrix(weights)

        for i in range(self.config.array_size - 1):
            self.step(weight_vec=weights[-1 - i])

        if data is not None:
            assert (
                len(data.shape) == 2
            ), "Data must be 2D array of shape (N, {self.config.array_size})"
            assert (
                data.shape[1] == self.config.array_size
            ), f"Data must be {self.config.array_size} wide"
            assert (
                data.shape[0] <= 2**self.config.accum_addr_width
            ), f"Not enough accumulator address bits to store all {data.shape[0]} data vectors"

            # Start streaming data into systolic array at the same time as last weights
            self.step(
                data_vec=data[0],
                weight_vec=weights[0],
                accum_addr=accum_addr,
                accum_mode=accum_mode,
                act_start=activation_enable,
                act_func=1 if activation_func == "relu" else 0,
            )

            for i in range(1, data.shape[0]):
                self.step(
                    data_vec=data[i],
                    accum_addr=accum_addr + i,
                    accum_mode=accum_mode,
                    act_start=activation_enable,
                    act_func=1 if activation_func == "relu" else 0,
                )
        # Flush pipeline
        if flush_pipeline:
            for _ in range(self.config.array_size + self.config.pipeline):
                self.step()

    def gemm(self, data: np.ndarray, weights: np.ndarray): ...

    def run_mlp(self, model: MLP, inputs: np.ndarray):
        """
        Run a MLP model on the accelerator. Predicts a single input (batch size of 1).

        Args:
            model: The MLP model to run.
            inputs: The input to the model (1d numpy array).

        Returns:
            np.ndarray: Predicted class probabilities
        """
        self.reset_output_trace()

        # Extract layer weights and biases
        weights_1 = model.fc1.weight.numpy(force=True)
        bias_1 = model.fc1.bias.numpy(force=True)
        weights_2 = model.fc2.weight.numpy(force=True)
        bias_2 = model.fc2.bias.numpy(force=True)

        # Add bias to first layer weights and 1 to activations
        W_aug, x_aug = bias_trick(weights_1, bias_1, inputs.flatten())

        for tile in generate_gemv_tiles(x_aug, W_aug, self.config.array_size):
            self.execute_instruction(
                weights=tile.matrix.T,
                data=tile.vector.reshape(1, -1),
                accum_addr=tile.index,
                accum_mode=not tile.first,
                activation_func="relu",
                activation_enable=tile.last,
                flush_pipeline=True,
            )

        self.step()
        self.step()
        self.step()

        x1 = np.array(self.output_trace).flatten()[: W_aug.shape[0]]
        self.reset_output_trace()

        W2_aug, x1_aug = bias_trick(weights_2, bias_2, x1)

        for tile in generate_gemv_tiles(x1_aug, W2_aug, self.config.array_size):
            self.execute_instruction(
                weights=tile.matrix.T,
                data=tile.vector.reshape(1, -1),
                accum_addr=tile.index,
                accum_mode=not tile.first,
                activation_func="relu",
                activation_enable=tile.last,
                flush_pipeline=True,
            )

        self.step()
        self.step()
        self.step()

        logits = np.array(self.output_trace).flatten()
        logprobs = softmax(logits)
        return logprobs

    def inspect_outputs(self) -> np.ndarray:
        """Get current output values converted to floating point"""
        return np.array(
            [
                self.config.activation_type(
                    binint=self.sim.inspect(f"out_{i}")
                ).decimal_approx
                for i in range(self.config.array_size)
            ]
        )

    def reset_output_trace(self):
        self.output_trace = []

    def run_mlp_batch(self, model: MLP, batch: np.ndarray) -> np.ndarray:
        """Run MLP inference on a batch of inputs."""
        self.reset_output_trace()

        # Extract weights and biases
        weights_1 = model.fc1.weight.numpy(force=True)
        bias_1 = model.fc1.bias.numpy(force=True)
        weights_2 = model.fc2.weight.numpy(force=True)
        bias_2 = model.fc2.bias.numpy(force=True)
        batch_size = batch.shape[0]

        tiles_done = 0
        tile_estimate = count_batch_gemm_tiles(
            128, 785, self.config.array_size
        ) + count_batch_gemm_tiles(10, 129, self.config.array_size)

        # print("\nInput shapes:")
        # print(f"batch: {batch.shape}")
        # print(f"weights_1: {weights_1.shape}")
        # print(f"bias_1: {bias_1.shape}")
        # print(f"weights_2: {weights_2.shape}")
        # print(f"bias_2: {bias_2.shape}")

        # First layer
        W1_aug, x_aug = bias_trick(weights_1, bias_1, batch)
        # print("\nAfter first bias trick:")
        # print(f"W1_aug shape: {W1_aug.shape}")
        # print(f"x_aug shape: {x_aug.shape}")

        # List to collect output chunks for first layer
        hidden_chunks = []

        # Process first layer tiles
        for tile in generate_batch_gemm_tiles(W1_aug, x_aug.T, self.config.array_size):
            self.execute_instruction(
                weights=tile.weight_tile.T,
                data=tile.batch_tile.T,
                accum_addr=0,
                accum_mode=not tile.first,
                activation_func="relu",
                activation_enable=tile.last,
                flush_pipeline=True,
            )

            if tile.last:
                self.step()
                self.step()
                results = np.array(self.output_trace)
                # print(f"\nFirst layer chunk shape: {results.shape}")
                hidden_chunks.append(results)
                self.reset_output_trace()

            tiles_done += 1
            print(f"Completed {tiles_done}/{tile_estimate} tiles", end="\r", flush=True)

        # Concatenate chunks and slice
        hidden_out = np.concatenate(hidden_chunks, axis=1)[:, : weights_1.shape[0]]
        # print(f"\nHidden layer output shape: {hidden_out.shape}")

        # Second layer
        W2_aug, h_aug = bias_trick(weights_2, bias_2, hidden_out)
        # print("\nAfter second bias trick:")
        # print(f"W2_aug shape: {W2_aug.shape}")
        # print(f"h_aug shape: {h_aug.shape}")

        # List to collect output chunks for second layer
        output_chunks = []

        # Process second layer tiles
        for tile in generate_batch_gemm_tiles(W2_aug, h_aug.T, self.config.array_size):
            self.execute_instruction(
                weights=tile.weight_tile.T,
                data=tile.batch_tile.T,
                accum_addr=0,
                accum_mode=not tile.first,
                activation_enable=tile.last,
                flush_pipeline=True,
            )

            if tile.last:
                self.step()
                self.step()
                results = np.array(self.output_trace)
                # print(f"\nSecond layer chunk shape: {results.shape}")
                # print(f"Second layer chunk content:\n{results}")
                output_chunks.append(results)
                self.reset_output_trace()
            print(f"Completed {tiles_done}/{tile_estimate} tiles", end="\r", flush=True)

        # print("\nNumber of output chunks:", len(output_chunks))
        # for i, chunk in enumerate(output_chunks):
        #     print(f"Chunk {i} shape: {chunk.shape}")

        # Concatenate chunks and slice
        final_out = np.concatenate(output_chunks, axis=1)[:, : weights_2.shape[0]]
        print(f"\nFinal output shape: {final_out.shape}")

        return np.apply_along_axis(softmax, 1, final_out)

    def inspect_accumulator_mem(self):
        return self.accelerator.inspect_accumulator_state(self.sim)
