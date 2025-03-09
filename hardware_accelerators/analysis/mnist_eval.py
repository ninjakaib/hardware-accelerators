# Evaluation function
from itertools import product
import os
import pandas as pd
from pyrtl import WireVector
import torch
from typing import Callable
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
import multiprocessing as mp
from tqdm.auto import tqdm
import time
import csv
from pathlib import Path
import traceback

from .config import (
    NN_TEST_MUL_FNS,
    NN_TEST_SYSTOLIC_ARRAY_SIZE,
    NN_TEST_ACCUM_ADDR_WIDTH,
    NN_TEST_WA_DTYPES,
    NN_TEST_BATCH_SIZE,
)
from hardware_accelerators.dtypes import *
from hardware_accelerators.simulation.accelerator import CompiledAcceleratorSimulator
from hardware_accelerators.rtllib.accelerator import CompiledAcceleratorConfig
from hardware_accelerators.rtllib.multipliers import *
from hardware_accelerators.nn import load_model
from ..simulation.accelerator import CompiledAcceleratorSimulator


def generate_test_configs(
    weight_act_dtypes: list[tuple[Type[BaseFloat], Type[BaseFloat]]],
    multiplier_fns: list[
        Callable[[WireVector, WireVector, type[BaseFloat]], WireVector]
    ],
):
    configs = []
    for dtypes, mult_fn in product(weight_act_dtypes, multiplier_fns):
        weight_type, act_type = dtypes
        config = CompiledAcceleratorConfig(
            array_size=NN_TEST_SYSTOLIC_ARRAY_SIZE,
            weight_type=weight_type,
            activation_type=act_type,
            multiplier=mult_fn,
            accum_addr_width=NN_TEST_ACCUM_ADDR_WIDTH,
        )
        configs.append(config)
    return configs


def evaluate_with_progress(
    config,
    dataset,
    batch_size,
    criterion=CrossEntropyLoss(),
    process_id=0,
):
    """Evaluate a model with progress tracking for the entire dataset"""
    # Define a complete result template with default values
    result = {
        "config": config.name,
        "weight_type": config.weight_type.__name__,
        "activation_type": config.activation_type.__name__,
        "multiplier": config.multiplier.__name__,
        "avg_loss": float("nan"),
        "accuracy": float("nan"),
        "total_time": 0,
        "batch_size": batch_size,
        "total_batches": 0,
        "total_samples": 0,
        "samples_per_second": 0,
        "error": None,  # Will be None for successful runs
    }

    try:
        start_time = time.time()

        # Load the appropriate model based on weight type
        if config.weight_type == Float32:
            model = load_model("./models/mlp_mnist_fp32.pth")
        else:
            model = load_model("./models/mlp_mnist_bf16.pth")

        # Create simulator
        sim = CompiledAcceleratorSimulator(config, model=model)

        if not sim.model_loaded:
            result["error"] = "No model loaded"
            return result

        correct = 0
        total = 0
        running_loss = 0.0

        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        total_batches = len(data_loader)
        result["total_batches"] = total_batches

        # Create a progress bar for this specific simulation
        desc = f"Config {config.name} ({config.weight_type.__name__}/{config.activation_type.__name__})"
        with tqdm(
            total=total_batches, desc=desc, position=process_id + 1, leave=False
        ) as pbar:
            for batch, labels in data_loader:
                batch_size_actual = batch.shape[0]
                batch = batch.reshape(batch_size_actual, -1).numpy()

                # Time the prediction
                outputs = sim.predict_batch(batch)

                loss = criterion(torch.tensor(outputs), labels)
                running_loss += loss.item()

                # Get predictions from the maximum value
                predicted = np.argmax(outputs, axis=1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # Update progress bar
                pbar.update(1)

        end_time = time.time()
        total_time = end_time - start_time

        # Update result with actual values
        result.update(
            {
                "avg_loss": running_loss / total_batches,
                "accuracy": 100.0 * correct / total,
                "total_time": total_time,
                "total_samples": total,
                "samples_per_second": total / total_time,
            }
        )

        return result

    except Exception as e:
        error_msg = (
            f"Error evaluating {config.name}: {str(e)}\n{traceback.format_exc()}"
        )
        print(error_msg)
        result["error"] = str(e)
        return result


def save_result(result, output_file):
    """Save a single result to CSV file"""
    file_exists = os.path.isfile(output_file)

    with open(output_file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(result.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(result)


def process_config(config, dataset, batch_size, output_file, process_id):
    """Process a single configuration and save results"""
    # Set process name for better monitoring
    mp.current_process().name = f"Sim-{config.name}"

    # print(f"Starting evaluation of {config.name}")
    result = evaluate_with_progress(config, dataset, batch_size, process_id=process_id)

    # Save result immediately
    save_result(result, output_file)

    print(
        f"Completed evaluation of {config.name}: Accuracy = {result.get('accuracy', 'ERROR'):.2f}%, "
        f"Time = {result.get('total_time', 0):.2f}s, "
        f"Speed = {result.get('samples_per_second', 0):.2f} samples/s"
    )
    return result


def main():
    # Create output directory
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "mnist_eval.csv"

    # Data transformation: convert images to tensor and normalize them
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    # Download MNIST test data
    test_dataset = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    configs = generate_test_configs(NN_TEST_WA_DTYPES, NN_TEST_MUL_FNS)

    # Create a multiprocessing pool
    num_processes = min(len(configs), mp.cpu_count() - 2)
    print(f"Using {num_processes} processes to evaluate {len(configs)} configurations")
    print(
        f"Each simulation will process the entire test dataset with batch size {NN_TEST_BATCH_SIZE}"
    )

    # Clear the screen and set up for progress bars
    print("\n\n")

    # Start the pool and process configurations
    with mp.Pool(processes=num_processes) as pool:
        # Create a list to track all tasks
        tasks = []

        # Submit all tasks
        for i, config in enumerate(configs):
            task = pool.apply_async(
                process_config,
                args=(config, test_dataset, NN_TEST_BATCH_SIZE, output_file, i),
            )
            tasks.append((config.name, task))

        # Set up the main progress bar at the top
        with tqdm(total=len(tasks), desc="Overall Progress", position=0) as pbar:
            completed = 0
            while completed < len(tasks):
                new_completed = sum(1 for _, task in tasks if task.ready())
                if new_completed > completed:
                    pbar.update(new_completed - completed)
                    completed = new_completed
                time.sleep(0.5)

        # Make sure all tasks are properly completed and collect results
        all_results = []
        for config_name, task in tasks:
            try:
                result = task.get()
                all_results.append(result)
            except Exception as e:
                print(f"Error in task {config_name}: {str(e)}")

    print(f"All evaluations complete. Results saved to {output_file}")

    # Create a summary DataFrame and display it
    if all_results:
        df = pd.DataFrame(all_results)
        print("\nSummary of Results (sorted by accuracy):")
        summary_cols = [
            "config",
            "weight_type",
            "activation_type",
            "multiplier",
            "accuracy",
            "total_time",
            "samples_per_second",
        ]
        print(df[summary_cols].sort_values("accuracy", ascending=False))

        print("\nSummary of Results (sorted by speed):")
        print(df[summary_cols].sort_values("samples_per_second", ascending=False))


if __name__ == "__main__":
    # Set start method for multiprocessing
    mp.set_start_method("spawn", force=True)  # Use 'spawn' for better compatibility
    main()
