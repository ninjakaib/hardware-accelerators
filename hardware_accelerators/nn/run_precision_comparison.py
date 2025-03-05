#!/usr/bin/env python3

import os
import argparse
from hardware_accelerators.nn.precision import train_model
from hardware_accelerators.nn.precision_eval import (
    compare_precision_inference,
    detailed_precision_comparison,
)


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Train and evaluate precision differences"
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Run detailed comparison with multiple batch sizes and trials",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=3,
        help="Number of trials to run for each batch size (only with --detailed)",
    )
    parser.add_argument(
        "--force-train",
        action="store_true",
        help="Force training a new model even if one exists",
    )
    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=[1, 16, 32, 64, 128, 256],
        help="Batch sizes to test (only with --detailed)",
    )
    args = parser.parse_args()

    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)

    # Path to save/load model
    model_path = "models/mlp_mnist_fp32.pth"

    # Check if model exists, train if not or if forced
    if not os.path.exists(model_path) or args.force_train:
        print("Training a new FP32 model...")
        # Train a model in FP32 precision
        train_model(
            precision="fp32",
            batch_size=64,
            hidden_size=128,
            num_epochs=5,
            learning_rate=0.001,
            optimizer_name="adam",
            model_save_path=model_path,
        )
    else:
        print(f"Using existing model at {model_path}")

    # Run comparison
    if args.detailed:
        print(
            f"Running detailed comparison with {args.trials} trials for each batch size..."
        )
        detailed_precision_comparison(
            model_path, trials=args.trials, batch_sizes=args.batch_sizes
        )
    else:
        # Run basic comparison
        compare_precision_inference(model_path)


if __name__ == "__main__":
    main()
