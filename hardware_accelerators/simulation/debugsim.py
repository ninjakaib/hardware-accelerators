import os
from pathlib import Path
from platformdirs import user_cache_dir
import pickle
from typing import Dict, Literal
import numpy as np
from pyrtl import (
    Input,
    Output,
    PyrtlError,
    SimulationTrace,
    FastSimulation,
)

from ..rtllib.systolic import SystolicArraySimState

from .cachedsim import CachedSimulation


from ..rtllib.accelerator import CompiledAccelerator, CompiledAcceleratorConfig

from ..nn.util import softmax

from ..nn.mlp import MLP

# from ..rtllib.multipliers import float_multiplier
from .matrix_utils import (
    bias_trick,
    count_batch_gemm_tiles,
    generate_batch_gemm_tiles,
    generate_gemv_tiles,
    permutate_weight_matrix,
    convert_array_dtype,
)
from typing import Optional, Literal
from IPython import get_ipython


def is_running_in_notebook():
    try:
        shell = get_ipython()
        if shell is not None:
            return True
        else:
            return False
    except NameError:
        return False


# Create progress bar
if is_running_in_notebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


class DebugAcceleratorSimulator:
    """Simulator for the accelerator that uses compiled simulation for speed."""

    def __init__(
        self,
        config: CompiledAcceleratorConfig,  # Now only accepts config
        model: MLP | None = None,
    ):
        """Initialize the simulator either with a config or from a saved binary.

        Args:
            config_or_path (Union[CompiledAcceleratorConfig, str]): Either an AcceleratorConfig object or a path to a saved binary
            model (Optional[MLP]): Neural network model to load into the accelerator. Defaults to None.
            recompile (bool): Whether to force recompilation of the binary even if path is provided. Defaults to False.
            cache_dir (Optional[Union[str, Path]]): Directory to save/load compiled binaries.

        Notes:
            If cache_dir is provided and recompile=False, loads the simulator state from the saved binary.
            If a config is provided or recompile=True, builds new simulator with the given configuration.
            If a model is provided, automatically loads it into the accelerator.
        """
        self.config = config
        self.model_loaded = False
        self.output_trace = []

        if model is not None:
            self.load_model(model)

        self.construct_hardware()
        self.reset_sim()

    def load_model(self, model: MLP):
        # Extract weights and biases
        self.model = model
        weights_1 = model.fc1.weight.numpy(force=True)
        bias_1 = model.fc1.bias.numpy(force=True)
        weights_2 = model.fc2.weight.numpy(force=True)
        bias_2 = model.fc2.bias.numpy(force=True)

        # Apply the bias trick
        W1_aug = bias_trick(weights=weights_1, bias=bias_1)
        W2_aug = bias_trick(weights=weights_2, bias=bias_2)

        self.W1_aug = convert_array_dtype(W1_aug, self.config.weight_type)
        self.W2_aug = convert_array_dtype(W2_aug, self.config.weight_type)
        self.input_dim = weights_1.shape[1]
        self.hidden_dim = weights_1.shape[0]
        self.output_dim = weights_2.shape[0]
        self.model_loaded = True

    def construct_hardware(self):
        """Construct the hardware for the accelerator."""
        # print(f"Constructing hardware for config {self.config.name}...")
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
        self._output_wires = [
            Output(self.config.activation_type.bitwidth(), f"out_{i}")
            for i in range(self.config.array_size)
        ]
        self.accelerator.connect_outputs(self._output_wires, Output(1, "output_valid"))

    def reset_sim(self):
        """Reset the simulation"""
        self.reset_output_trace()
        wires_to_track = self.accelerator.systolic_array.get_wires_to_track()
        self.simtracer = SimulationTrace(wires_to_track)
        self.sim = FastSimulation(tracer=self.simtracer)
        self.steps = 0
        self.systolic_history = []

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

    def _step(
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
                inputs[f"weight_in_{i}"] = int(value)
        if data_vec is not None:
            if len(data_vec) != self.config.array_size:
                raise ValueError(f"Data vector length must be {self.config.array_size}")
            inputs["data_enable"] = 1
            for i, value in enumerate(data_vec):
                inputs[f"data_in_{i}"] = int(value)

        self.sim.step(inputs)
        self.steps += 1
        if self.sim.inspect("output_valid") == 1:
            self.output_trace.append(self.inspect_outputs())

        if kwargs.pop("inspect_systolic", False):
            self.systolic_history.append(
                self.inspect_systolic_state(kwargs.pop("step_name"))
            )

    # TODO: for debugging lmul, remove later
    def inspect_systolic_state(self, step_name) -> SystolicArraySimState:
        """Get current output values converted to floating point"""
        stepstr = f"{self.steps}: {step_name}"
        return self.accelerator.systolic_array.get_state(self.sim, stepstr)

    def execute_instruction(
        self,
        weights: np.ndarray,
        data: np.ndarray,
        accum_addr: int = 0,
        accum_mode: int = 0,  # 0=overwrite, 1=accumulate
        activation_enable: bool = False,  # 0=nothing, 1=enable
        activation_func: Optional[Literal["relu"]] = None,
        flush_pipeline: bool = True,
        **kwargs,
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
            self._step(
                weight_vec=weights[-1 - i],
                step_name=f"LOAD_WEIGHTS_{i}/{self.config.array_size-1}",
                **kwargs,
            )

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
            self._step(
                data_vec=data[0],
                weight_vec=weights[0],
                accum_addr=accum_addr,
                accum_mode=accum_mode,
                act_start=activation_enable,
                act_func=1 if activation_func == "relu" else 0,
                step_name="LOAD_DATA_0 & LOAD_LAST_WEIGHT",
                **kwargs,
            )

            for i in range(1, data.shape[0]):
                self._step(
                    data_vec=data[i],
                    accum_addr=accum_addr + i,
                    accum_mode=accum_mode,
                    act_start=activation_enable,
                    act_func=1 if activation_func == "relu" else 0,
                    step_name=f"LOAD_DATA_{i}/{data.shape[0]-1}",
                    **kwargs,
                )
        # Flush pipeline
        if flush_pipeline:
            for _ in range(self.config.array_size + self.config.pipeline_pe):
                self._step(step_name="FLUSH_PIPELINE", **kwargs)

    def predict(self, input: np.ndarray):
        """
        Run a MLP model on the accelerator. Predicts a single input (batch size of 1).

        Args:
            inputs: The input to the model (1d numpy array).

        Returns:
            np.ndarray: Predicted class probabilities
        """
        if not self.model_loaded:
            raise RuntimeError(
                "Model not loaded. Call load_model() first or pass a model to the constructor."
            )

        self.reset_output_trace()

        # First layer
        x_aug = convert_array_dtype(bias_trick(x=input), self.config.activation_type)

        for tile in generate_gemv_tiles(x_aug, self.W1_aug, self.config.array_size):
            self.execute_instruction(
                weights=tile.matrix.T,
                data=tile.vector.reshape(1, -1),
                accum_addr=tile.index,
                accum_mode=not tile.first,
                activation_func="relu",
                activation_enable=tile.last,
                flush_pipeline=True,
            )

        self._step()
        self._step()
        self._step()

        hidden_out = np.array(self.output_trace).flatten()[: self.hidden_dim]
        self.reset_output_trace()

        h_aug = convert_array_dtype(
            bias_trick(x=hidden_out), self.config.activation_type
        )

        for tile in generate_gemv_tiles(h_aug, self.W2_aug, self.config.array_size):
            self.execute_instruction(
                weights=tile.matrix.T,
                data=tile.vector.reshape(1, -1),
                accum_addr=tile.index,
                accum_mode=not tile.first,
                activation_enable=tile.last,
                flush_pipeline=True,
            )

        self._step()
        self._step()
        self._step()

        logits = np.array(self.output_trace).flatten()[: self.output_dim]
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

    def predict_batch(
        self,
        batch: np.ndarray,
        apply_softmax: bool = True,
        print_progress: bool = False,
    ) -> np.ndarray:
        """Run MLP inference on a batch of inputs."""

        if not self.model_loaded:
            raise RuntimeError(
                "Model not loaded. Call load_model() first or pass a model to the constructor."
            )

        self.reset_output_trace()

        tiles_done = 0
        tile_estimate = count_batch_gemm_tiles(
            self.hidden_dim, self.input_dim + 1, self.config.array_size
        ) + count_batch_gemm_tiles(
            self.output_dim, self.hidden_dim + 1, self.config.array_size
        )

        # First layer
        x_aug = convert_array_dtype(bias_trick(x=batch), self.config.activation_type)

        # List to collect output chunks for first layer
        hidden_chunks = []

        # Process first layer tiles
        for tile in generate_batch_gemm_tiles(
            self.W1_aug, x_aug.T, self.config.array_size
        ):
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
                self._step()
                self._step()
                self._step()
                results = np.array(self.output_trace)
                hidden_chunks.append(results)
                self.reset_output_trace()

            tiles_done += 1
            if print_progress:
                print(
                    f"Completed {tiles_done}/{tile_estimate} tiles",
                    end="\r",
                    flush=True,
                )

        # Concatenate chunks and slice
        hidden_out = np.concatenate(hidden_chunks, axis=1)[:, : self.hidden_dim]

        # Second layer
        h_aug = convert_array_dtype(
            bias_trick(x=hidden_out), self.config.activation_type
        )

        # List to collect output chunks for second layer
        output_chunks = []

        # Process second layer tiles
        for tile in generate_batch_gemm_tiles(
            self.W2_aug, h_aug.T, self.config.array_size
        ):
            self.execute_instruction(
                weights=tile.weight_tile.T,
                data=tile.batch_tile.T,
                accum_addr=0,
                accum_mode=not tile.first,
                activation_enable=tile.last,
                flush_pipeline=True,
            )

            if tile.last:
                self._step()
                self._step()
                self._step()
                results = np.array(self.output_trace)
                output_chunks.append(results)
                self.reset_output_trace()

            tiles_done += 1
            if print_progress:
                print(
                    f"Completed {tiles_done}/{tile_estimate} tiles",
                    end="\r",
                    flush=True,
                )

        # Concatenate chunks and slice
        final_out = np.concatenate(output_chunks, axis=1)[:, : self.output_dim]
        if apply_softmax:
            return np.apply_along_axis(softmax, 1, final_out)
        else:
            return {"hidden_out": hidden_out, "final_out": final_out}

    def predict_batch_traced(self, batch: np.ndarray):
        """Run MLP inference on a batch of inputs with progress tracking."""
        if not self.model_loaded:
            raise RuntimeError(
                "Model not loaded. Call load_model() first or pass a model to the constructor."
            )

        self.reset_output_trace()
        sim_results = {}
        numpy_results = self.predict_batch_numpy(batch)

        # Calculate total number of tiles
        total_tiles = count_batch_gemm_tiles(
            self.hidden_dim, self.input_dim + 1, self.config.array_size
        ) + count_batch_gemm_tiles(
            self.output_dim, self.hidden_dim + 1, self.config.array_size
        )

        pbar = tqdm(total=total_tiles, desc="Processing tiles")

        # First layer
        x_aug = convert_array_dtype(bias_trick(x=batch), self.config.activation_type)

        # List to collect output chunks for first layer
        hidden_chunks = []

        # Process first layer tiles
        for tile in generate_batch_gemm_tiles(
            self.W1_aug, x_aug.T, self.config.array_size
        ):
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
                self._step()
                self._step()
                self._step()
                results = np.array(self.output_trace)
                hidden_chunks.append(results)
                self.reset_output_trace()

            # Update progress bar
            pbar.update(1)

        # Concatenate chunks and slice
        hidden_out = np.concatenate(hidden_chunks, axis=1)[:, : self.hidden_dim]

        self.systolic_outs = []

        # Second layer
        h_aug = convert_array_dtype(
            bias_trick(x=hidden_out), self.config.activation_type
        )

        # List to collect output chunks for second layer
        output_chunks = []

        # Process second layer tiles
        for tile in generate_batch_gemm_tiles(
            self.W2_aug, h_aug.T, self.config.array_size
        ):
            self.execute_instruction(
                weights=tile.weight_tile.T,
                data=tile.batch_tile.T,
                accum_addr=0,
                accum_mode=not tile.first,
                activation_enable=tile.last,
                flush_pipeline=True,
                inspect_systolic=True,
            )

            if tile.last:
                self._step(inspect_systolic=True, step_name="FLUSH_LAST_TILE")
                self._step(inspect_systolic=True, step_name="FLUSH_LAST_TILE")
                self._step(inspect_systolic=True, step_name="FLUSH_LAST_TILE")
                results = np.array(self.output_trace)
                output_chunks.append(results)
                self.reset_output_trace()

            # Update progress bar
            pbar.update(1)

        # Close progress bar
        pbar.close()

        # Concatenate chunks and slice
        final_out = np.concatenate(output_chunks, axis=1)[:, : self.output_dim]
        logprogs = np.apply_along_axis(softmax, 1, final_out)
        predictions = np.argmax(logprogs, axis=1)

        sim_results = {
            "a1": hidden_out,
            "h_out": final_out,
            "logprogs": logprogs,
            "predictions": predictions,
        }

        return sim_results, numpy_results

    def predict_batch_numpy(self, batch: np.ndarray):
        # Vectorized standard NumPy implementation
        weights_1 = self.model.fc1.weight.numpy(force=True)
        bias_1 = self.model.fc1.bias.numpy(force=True)
        weights_2 = self.model.fc2.weight.numpy(force=True)
        bias_2 = self.model.fc2.bias.numpy(force=True)

        h1_numpy = batch @ weights_1.T + bias_1
        a1_numpy = np.maximum(0, h1_numpy)
        h_out_numpy = a1_numpy @ weights_2.T + bias_2

        logprogs = np.apply_along_axis(softmax, 1, h_out_numpy)
        predictions = np.argmax(logprogs, axis=1)

        return {
            "h1": h1_numpy,  # hidden layer 1 output before relu
            "a1": a1_numpy,  # relu output
            "h_out": h_out_numpy,  # hidden layer 2 output before softmax
            "logprogs": logprogs,  # softmax output
            "predictions": predictions,  # argmax of softmax output
        }

    def gemm(self, matrixA, matrixB):
        a = convert_array_dtype(matrixA, self.config.activation_type)
        b = convert_array_dtype(matrixB, self.config.weight_type)
        output_chunks = []
        for tile in generate_batch_gemm_tiles(b, a.T, self.config.array_size):
            self.execute_instruction(
                weights=tile.weight_tile.T,
                data=tile.batch_tile.T,
                accum_addr=0,
                accum_mode=not tile.first,
                activation_enable=tile.last,
                flush_pipeline=True,
            )

            if tile.last:
                self._step()
                self._step()
                self._step()
                results = np.array(self.output_trace)
                output_chunks.append(results)
                self.reset_output_trace()

        # Concatenate chunks and slice
        result = np.concatenate(output_chunks, axis=1)[:, : a.shape[0]]
        return result

    def inspect_accumulator_mem(self):
        return self.accelerator.inspect_accumulator_state(self.sim)
