import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import time
from pathlib import Path

from .precision import MLP
from .util import get_pytorch_device


def load_mnist_data(batch_size=100):
    """Load MNIST test dataset"""
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    test_dataset = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader


def create_model(precision):
    """Create MLP model with specified precision"""
    input_size = 28 * 28  # MNIST images are 28x28
    hidden_size = 128
    num_classes = 10
    device = get_pytorch_device()

    model = MLP(input_size, hidden_size, num_classes).to(device)

    # Convert model to target precision
    if precision == "fp16":
        model = model.to(torch.float16)
    elif precision == "bf16":
        model = model.to(torch.bfloat16)
    elif precision == "fp32":
        model = model.to(torch.float32)

    return model, device


def load_model_weights(model, model_path):
    """Load model weights from checkpoint"""
    model.load_state_dict(torch.load(model_path))
    return model


def evaluate_model(model, test_loader, device, precision):
    """Evaluate model accuracy and inference time"""
    model.eval()
    correct = 0
    total = 0

    # For measuring inference time
    start_time = time.time()

    with torch.no_grad():
        for data, target in test_loader:
            # Convert input to specified precision
            if precision == "fp16":
                data = data.half()
            elif precision == "bf16":
                data = data.to(torch.bfloat16)

            data, target = data.to(device), target.to(device)

            # Forward pass
            outputs = model(data)

            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    inference_time = time.time() - start_time
    accuracy = 100.0 * correct / total

    return {
        "accuracy": accuracy,
        "inference_time": inference_time,
        "correct": correct,
        "total": total,
    }


def compare_precision_inference(fp32_model_path):
    """Compare FP32 trained model inference in different precisions"""
    print("\nEvaluating FP32-trained model inference with different precisions")
    print("-" * 80)

    # Load test data
    test_loader = load_mnist_data()

    # Verify the model file exists
    model_path = Path(fp32_model_path)
    if not model_path.exists():
        print(f"Error: Model file {fp32_model_path} not found!")
        return

    # Create models in different precisions
    precisions = ["fp32", "bf16"]
    results = {}

    for precision in precisions:
        print(f"\nTesting inference in {precision.upper()} mode...")

        # Create model in specified precision
        model, device = create_model(precision)

        # Load weights from the FP32-trained model
        # When loading to a BF16 model, the weights will be automatically cast
        model = load_model_weights(model, fp32_model_path)

        # Evaluate model
        results[precision] = evaluate_model(model, test_loader, device, precision)

        # Print results
        print(f"Accuracy: {results[precision]['accuracy']:.2f}%")
        print(f"Inference time: {results[precision]['inference_time']:.4f} seconds")

    # Calculate and print comparison metrics
    print("\nComparison Summary")
    print("-" * 80)

    fp32_results = results["fp32"]
    bf16_results = results["bf16"]

    acc_diff = bf16_results["accuracy"] - fp32_results["accuracy"]
    time_ratio = fp32_results["inference_time"] / bf16_results["inference_time"]

    print(f"Accuracy drop from FP32 to BF16: {acc_diff:.2f}%")
    print(f"Inference speedup with BF16: {time_ratio:.2f}x")

    return results


def detailed_precision_comparison(
    fp32_model_path, trials=3, batch_sizes=[1, 16, 32, 64, 128, 256]
):
    """Run detailed comparison with multiple batch sizes and trials"""
    print("\nDetailed Precision Comparison")
    print("=" * 80)

    # Verify the model file exists
    model_path = Path(fp32_model_path)
    if not model_path.exists():
        print(f"Error: Model file {fp32_model_path} not found!")
        return

    precisions = ["fp32", "bf16"]
    all_results = {}

    for precision in precisions:
        all_results[precision] = {}

        print(f"\nEvaluating {precision.upper()} precision")
        print("-" * 60)

        # Create model in specified precision only once
        model, device = create_model(precision)
        model = load_model_weights(model, fp32_model_path)

        # Warm up the GPU/CPU
        print("Warming up...")
        dummy_loader = load_mnist_data(batch_size=64)
        with torch.no_grad():
            for data, _ in dummy_loader:
                if precision == "bf16":
                    data = data.to(torch.bfloat16)
                elif precision == "fp16":
                    data = data.half()
                data = data.to(device)
                _ = model(data)
                break

        # Run trials for different batch sizes
        for batch_size in batch_sizes:
            print(f"\n  Batch size: {batch_size}")
            test_loader = load_mnist_data(batch_size=batch_size)

            batch_results = {"accuracy": [], "inference_time": []}

            for trial in range(trials):
                print(f"    Trial {trial+1}/{trials}...", end="", flush=True)
                result = evaluate_model(model, test_loader, device, precision)
                batch_results["accuracy"].append(result["accuracy"])
                batch_results["inference_time"].append(result["inference_time"])
                print(
                    f" done. Time: {result['inference_time']:.4f}s, Accuracy: {result['accuracy']:.2f}%"
                )

            # Calculate averages
            avg_accuracy = sum(batch_results["accuracy"]) / len(
                batch_results["accuracy"]
            )
            avg_time = sum(batch_results["inference_time"]) / len(
                batch_results["inference_time"]
            )

            all_results[precision][batch_size] = {
                "avg_accuracy": avg_accuracy,
                "avg_inference_time": avg_time,
                "trials": batch_results,
            }

            print(f"    Average: {avg_time:.4f}s, Accuracy: {avg_accuracy:.2f}%")

    # Print comparison table
    print("\nComparison Results")
    print("=" * 80)
    print(
        f"{'Batch Size':^10} | {'FP32 Acc':^10} | {'BF16 Acc':^10} | {'Acc Diff':^10} | {'FP32 Time':^10} | {'BF16 Time':^10} | {'Speedup':^10}"
    )
    print("-" * 80)

    for batch_size in batch_sizes:
        fp32_acc = all_results["fp32"][batch_size]["avg_accuracy"]
        bf16_acc = all_results["bf16"][batch_size]["avg_accuracy"]
        acc_diff = bf16_acc - fp32_acc

        fp32_time = all_results["fp32"][batch_size]["avg_inference_time"]
        bf16_time = all_results["bf16"][batch_size]["avg_inference_time"]
        speedup = fp32_time / bf16_time

        print(
            f"{batch_size:^10} | {fp32_acc:^10.2f} | {bf16_acc:^10.2f} | {acc_diff:^10.2f} | {fp32_time:^10.4f} | {bf16_time:^10.4f} | {speedup:^10.2f}x"
        )

    # Calculate and print overall averages
    avg_acc_diff = sum(
        all_results["bf16"][bs]["avg_accuracy"]
        - all_results["fp32"][bs]["avg_accuracy"]
        for bs in batch_sizes
    ) / len(batch_sizes)
    avg_speedup = sum(
        all_results["fp32"][bs]["avg_inference_time"]
        / all_results["bf16"][bs]["avg_inference_time"]
        for bs in batch_sizes
    ) / len(batch_sizes)

    print("-" * 80)
    print(f"Average accuracy difference across all batch sizes: {avg_acc_diff:.2f}%")
    print(f"Average speedup across all batch sizes: {avg_speedup:.2f}x")

    return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Compare model inference with FP32 vs BF16"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/mlp_mnist_fp32.pth",
        help="Path to FP32 trained model weights",
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Run detailed comparison with multiple batch sizes",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=3,
        help="Number of trials to run (only with --detailed)",
    )

    args = parser.parse_args()

    if args.detailed:
        detailed_precision_comparison(args.model_path, trials=args.trials)
    else:
        compare_precision_inference(args.model_path)
