import os
import time
import multiprocessing
import pyrtl
from tqdm import tqdm
from functools import partial

from .simulation import CompiledAcceleratorSimulator

from .rtllib import float_multiplier, lmul_fast, float_adder
from .rtllib.accelerator import (
    CompiledAccelerator,
    CompiledAcceleratorConfig,
)
from .dtypes import BaseFloat, Float32, Float16, BF16, Float8
from typing import Iterator, Type, List, Callable
from itertools import product


def generate_accelerator_configs(
    array_size: int = 8,
    dtypes: List[Type[BaseFloat]] | None = None,
    multipliers: List[Callable] | None = None,
    **kwargs,
) -> Iterator[CompiledAcceleratorConfig]:
    """
    Generate all valid CompiledAcceleratorConfig combinations.

    Args:
        array_size: Size of the systolic array
        dtypes: List of data types to consider. Defaults to [Float8, BF16, FP16, FP32]
        multipliers: List of multiplier functions. Defaults to [float_multiplier, lmul]

    Yields:
        Valid CompiledAcceleratorConfig objects

    Restrictions:
        1. The activation_type must be greater than or equal to the weight_type in terms of bitwidth.
        2. 16-bit float types (BF16, FP16) should not be combined with each other.
           They should only pair with themselves or with FP32.
    """
    if dtypes is None:
        dtypes = [Float8, BF16, Float16, Float32]

    if multipliers is None:
        multipliers = [float_multiplier, lmul_fast]

    # Sort dtypes by bitwidth for easier comparison
    dtype_bitwidths = {dtype: dtype.bitwidth() for dtype in dtypes}
    sorted_dtypes = sorted(dtypes, key=lambda d: dtype_bitwidths[d])

    # Identify 16-bit float types
    bit16_float_types = [dtype for dtype in dtypes if dtype_bitwidths[dtype] == 16]

    # Generate all combinations
    for multiplier in multipliers:
        for weight_type in sorted_dtypes:
            # Find valid activation types based on bitwidth
            valid_activation_types = [
                dtype
                for dtype in sorted_dtypes
                if dtype_bitwidths[dtype] >= dtype_bitwidths[weight_type]
            ]

            for activation_type in valid_activation_types:
                # Skip invalid combinations of 16-bit float types
                if (
                    weight_type in bit16_float_types
                    and activation_type in bit16_float_types
                    and weight_type != activation_type
                ):
                    continue

                yield CompiledAcceleratorConfig(
                    array_size=array_size,
                    activation_type=activation_type,
                    weight_type=weight_type,
                    multiplier=multiplier,
                    **kwargs,
                )


def compile_and_save_simulator(config):
    """Compile and save a simulator for a given configuration.

    Args:
        config: The CompiledAcceleratorConfig to use

    Returns:
        Tuple of (config, success, time_taken)
    """
    start_time = time.time()

    try:
        # Create the simulator
        with pyrtl.temp_working_block():
            CompiledAcceleratorSimulator(config)

        end_time = time.time()
        return (config, True, end_time - start_time)

    except Exception as e:
        end_time = time.time()
        print(f"Error compiling {config}: {str(e)}")
        return (config, False, end_time - start_time)


def compile_all_simulators(configs=None, max_workers=None):
    """Compile and save simulators for all configurations using multiprocessing.

    Args:
        configs: List of configurations to compile. If None, generates all valid configs.
        base_dir: Base directory to save simulations
        max_workers: Maximum number of worker processes. If None, uses CPU count.

    Returns:
        List of results (config, success, time_taken)
    """
    if configs is None:
        configs = list(generate_accelerator_configs())

    if max_workers is None:
        max_workers = multiprocessing.cpu_count()

    print(f"Compiling {len(configs)} configurations using {max_workers} workers")

    # Create a partial function with the base_dir parameter
    compile_func = partial(compile_and_save_simulator)

    # Use multiprocessing to compile all configurations
    with multiprocessing.Pool(processes=max_workers) as pool:
        # Use tqdm to show progress
        results = list(
            tqdm(
                pool.imap(compile_func, configs),
                total=len(configs),
                desc="Compiling simulators",
            )
        )

    # Print summary
    successful = [r for r in results if r[1]]
    failed = [r for r in results if not r[1]]

    print(f"\nCompilation complete:")
    print(f"  Total: {len(results)}")
    print(f"  Successful: {len(successful)}")
    print(f"  Failed: {len(failed)}")

    if successful:
        avg_time = sum(r[2] for r in successful) / len(successful)
        print(f"  Average compilation time: {avg_time:.2f} seconds")

    return results


if __name__ == "__main__":
    # Generate all valid configurations
    all_configs = list(generate_accelerator_configs())
    print(f"Generated {len(all_configs)} configs")

    # Compile and save simulators for all configurations
    results = compile_all_simulators(all_configs)

    # Print details of failed compilations if any
    failed = [r for r in results if not r[1]]
    if failed:
        print("\nFailed compilations:")
        for config, _, _ in failed:
            print(config.name)
