# Example usage
from tqdm import tqdm
import numpy as np
from typing import Type, Tuple
from hardware_accelerators.dtypes import BaseFloat, BF16, Float32
from .rtllib.utils.lmul_utils import get_combined_offset
from hardware_accelerators.nn import load_model, softmax
from pyrtl import *
import numpy as np
from typing import Type
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


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
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


def batch_loader(batch_size):
    loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    for batch, labels in tqdm(iter(loader)):
        yield batch.reshape(batch_size, -1).numpy(), labels.numpy()


def get_activation():
    image, _ = next(iter(test_loader))
    image = image.detach().numpy().reshape(-1)
    return image


class OptimizedLmul:
    """Optimized implementation of LMUL using pure integer arithmetic"""

    def __init__(self, dtype: Type[BaseFloat]):
        self.dtype = dtype
        self.lmul_offset = {
            BF16: get_combined_offset(8, 7, True),
            Float32: get_combined_offset(8, 23, True),
        }
        self.bitmask = {
            BF16: 0b0111111111111111,
            Float32: 0b01111111111111111111111111111111,
        }
        self.bitwidth = dtype.bitwidth()
        self.mantissa_bits = dtype.mantissa_bits()

    def multiply(self, bin_a: int, bin_b: int) -> int:
        """Multiply two numbers in binint representation using LMUL algorithm"""
        # Extract sign bit
        sign = (bin_a >> (self.bitwidth - 1)) ^ (bin_b >> (self.bitwidth - 1))

        # Clear sign bits
        bin_a &= self.bitmask[self.dtype]
        bin_b &= self.bitmask[self.dtype]

        # Check for zero exponents (denormals or zero)
        if bin_a >> self.mantissa_bits == 0 or bin_b >> self.mantissa_bits == 0:
            return 0

        # Apply LMUL algorithm
        binint = (bin_a + bin_b + self.lmul_offset[self.dtype]) & self.bitmask[
            self.dtype
        ]

        # Set sign bit
        binint |= sign << (self.bitwidth - 1)

        return binint


def convert_to_binint(data: np.ndarray, dtype: Type[BaseFloat]) -> np.ndarray:
    """Convert numpy array of floats to array of binint representations"""
    # Create a vectorized function to convert each element
    vectorized_convert = np.vectorize(lambda x: dtype(x).binint)
    return vectorized_convert(data)


def convert_from_binint(data: np.ndarray, dtype: Type[BaseFloat]) -> np.ndarray:
    """Convert numpy array of binint representations back to floats"""
    # Create a vectorized function to convert each element
    vectorized_convert = np.vectorize(lambda x: float(dtype(binint=x)))
    return vectorized_convert(data)


def relu_binint(x: np.ndarray, dtype: Type[BaseFloat]) -> np.ndarray:
    """Apply ReLU to binint values by checking sign bit"""
    # For floating point in binint representation, negative numbers have the highest bit set
    sign_mask = 1 << (dtype.bitwidth() - 1)
    return np.where((x & sign_mask) == 0, x, 0)


def cross_entropy_loss(predictions: np.ndarray, targets: np.ndarray) -> float:
    """
    Calculate cross-entropy loss for classification

    Args:
        predictions: Model output probabilities [batch_size, num_classes]
        targets: Ground truth labels [batch_size]

    Returns:
        Average cross-entropy loss
    """
    batch_size = predictions.shape[0]

    # Create one-hot encoding of targets
    num_classes = predictions.shape[1]
    one_hot_targets = np.zeros((batch_size, num_classes))
    for i in range(batch_size):
        one_hot_targets[i, targets[i]] = 1

    # Add small epsilon to prevent log(0)
    epsilon = 1e-15
    predictions = np.clip(predictions, epsilon, 1 - epsilon)

    # Calculate cross-entropy loss
    losses = -np.sum(one_hot_targets * np.log(predictions), axis=1)

    # Return average loss
    return np.mean(losses)


def optimized_mlp_inference(
    inputs_batch: np.ndarray, model_weights: dict, dtype: Type[BaseFloat]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Optimized MLP inference using integer arithmetic for batched inputs

    Args:
        inputs_batch: Batch of input vectors [batch_size, input_dim]
        model_weights: Dictionary containing model weights and biases
        dtype: The floating-point data type to use

    Returns:
        Tuple of (predicted_classes, output_probabilities)
    """
    # Initialize the optimized LMUL
    lmul = OptimizedLmul(dtype)

    # Pre-convert all weights and biases to binint representation
    fc1_weight_bin = convert_to_binint(model_weights["fc1_weight"], dtype)
    fc1_bias_bin = convert_to_binint(model_weights["fc1_bias"], dtype)
    fc2_weight_bin = convert_to_binint(model_weights["fc2_weight"], dtype)
    fc2_bias_bin = convert_to_binint(model_weights["fc2_bias"], dtype)

    # Convert inputs to binint
    inputs_bin = convert_to_binint(inputs_batch, dtype)

    batch_size = inputs_batch.shape[0]
    hidden_size = fc1_weight_bin.shape[0]
    output_size = fc2_weight_bin.shape[0]
    input_size = inputs_batch.shape[1]

    # First layer: matrix multiplication + bias + ReLU
    h1_bin = np.zeros((batch_size, hidden_size), dtype=np.int64)

    # Perform matrix multiplication with LMUL
    for b in range(batch_size):
        for i in range(hidden_size):
            acc = 0  # Accumulate in regular float for now
            for j in range(input_size):
                # Multiply using optimized LMUL
                prod = lmul.multiply(fc1_weight_bin[i, j], inputs_bin[b, j])

                # Convert product back to float for accumulation
                # In a real hardware implementation, this would be a floating-point addition
                prod_float = float(dtype(binint=prod))
                acc += prod_float

            # Convert accumulated result back to binint
            h1_bin[b, i] = dtype(acc).binint

    # Add bias (in binint space, this is still a floating-point addition)
    h1_with_bias_bin = np.zeros_like(h1_bin)
    for b in range(batch_size):
        for i in range(hidden_size):
            # Convert to float, add, then convert back to binint
            h1_float = float(dtype(binint=h1_bin[b, i]))
            bias_float = float(dtype(binint=fc1_bias_bin[i]))
            h1_with_bias_bin[b, i] = dtype(h1_float + bias_float).binint

    # Apply ReLU in binint space
    a1_bin = relu_binint(h1_with_bias_bin, dtype)

    # Second layer: matrix multiplication + bias
    h_out_bin = np.zeros((batch_size, output_size), dtype=np.int64)

    # Perform matrix multiplication with LMUL
    for b in range(batch_size):
        for i in range(output_size):
            acc = 0  # Accumulate in regular float for now
            for j in range(hidden_size):
                # Multiply using optimized LMUL
                prod = lmul.multiply(fc2_weight_bin[i, j], a1_bin[b, j])

                # Convert product back to float for accumulation
                prod_float = float(dtype(binint=prod))
                acc += prod_float

            # Convert accumulated result back to binint
            h_out_bin[b, i] = dtype(acc).binint

    # Add bias
    h_out_with_bias_bin = np.zeros_like(h_out_bin)
    for b in range(batch_size):
        for i in range(output_size):
            # Convert to float, add, then convert back to binint
            h_out_float = float(dtype(binint=h_out_bin[b, i]))
            bias_float = float(dtype(binint=fc2_bias_bin[i]))
            h_out_with_bias_bin[b, i] = dtype(h_out_float + bias_float).binint

    # Convert final layer output back to floats for softmax
    h_out_float = convert_from_binint(h_out_with_bias_bin, dtype)

    # Apply softmax (in float space)
    output_probs = np.zeros_like(h_out_float)
    for b in range(batch_size):
        exp_x = np.exp(
            h_out_float[b] - np.max(h_out_float[b])
        )  # For numerical stability
        output_probs[b] = exp_x / exp_x.sum()

    # Get predicted classes
    predicted_classes = np.argmax(output_probs, axis=1)

    return predicted_classes, output_probs


if __name__ == "__main__":
    from hardware_accelerators.dtypes import BF16
    from hardware_accelerators.nn import load_model
    import time

    # Load the model
    model = load_model("models/mlp_mnist_bf16.pth", "cpu")

    # Extract weights and biases
    fc1_weight = model.fc1.weight.data.numpy()
    fc1_bias = model.fc1.bias.data.numpy()
    fc2_weight = model.fc2.weight.data.numpy()
    fc2_bias = model.fc2.bias.data.numpy()

    # Store weights in a dictionary
    model_weights = {
        "fc1_weight": fc1_weight,
        "fc1_bias": fc1_bias,
        "fc2_weight": fc2_weight,
        "fc2_bias": fc2_bias,
    }

    # Get batch of input data
    batch_size = 100

    # Run evaluation on the test dataset
    total_correct = 0
    total_samples = 0
    total_loss = 0.0
    num_batches = 0

    print(f"Evaluating model with BF16 precision and LMUL...")

    start_time = time.time()
    for inputs_batch, labels in batch_loader(batch_size):
        # Run inference
        predicted_classes, output_probs = optimized_mlp_inference(
            inputs_batch, model_weights, BF16
        )

        # Calculate accuracy
        correct = np.sum(predicted_classes == labels)
        total_correct += correct
        total_samples += len(labels)

        # Calculate loss
        batch_loss = cross_entropy_loss(output_probs, labels)
        total_loss += batch_loss
        num_batches += 1

        # Print batch results
        print(
            f"Batch {num_batches}: Accuracy = {correct}/{len(labels)} ({correct/len(labels)*100:.2f}%), Loss = {batch_loss:.4f}"
        )
        break

    # Calculate final metrics
    total_time = time.time() - start_time
    avg_accuracy = total_correct / total_samples
    avg_loss = total_loss / num_batches

    print("\nEvaluation Results:")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average inference time per batch: {total_time/num_batches:.4f} seconds")
    print(f"Final accuracy: {total_correct}/{total_samples} ({avg_accuracy*100:.2f}%)")
    print(f"Average loss: {avg_loss:.4f}")

    # Show a few examples
    # print("\nSample comparisons:")
    # for i in range(min(5, batch_size)):
    #     print(
    #         f"Sample {i}: NumPy predicted {numpy_predictions[i]}, Optimized predicted {predicted_classes[i]}"
    #     )
