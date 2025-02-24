from dataclasses import dataclass
import numpy as np
from typing import List, Tuple, Generator, Callable, Type
import matplotlib.pyplot as plt
from torch import Tensor

from ..dtypes.base import BaseFloat


def print_arrays(*args, **kwargs):
    # print the name of the array and its content with numpy print options to set precision to 3
    for arr in args:
        if isinstance(arr, np.ndarray):
            print(f"Array of shape {arr.shape}:")
            print(np.array2string(arr, precision=3, suppress_small=True))
            print()
        elif isinstance(arr, Tensor):
            print(f"Tensor of shape {arr.shape}:")
            print(
                np.array2string(
                    arr.float().numpy(force=True), precision=3, suppress_small=True
                )
            )
            print()
    for name, arr in kwargs.items():
        if isinstance(arr, np.ndarray):
            print(f"{name} {arr.shape}:")
            print(np.array2string(arr, precision=3, suppress_small=True))
            print()
        elif isinstance(arr, Tensor):
            print(f"{name} {arr.shape}:")
            print(
                np.array2string(
                    arr.float().numpy(force=True), precision=3, suppress_small=True
                )
            )
            print()


def convert_array_dtype(
    arr: np.ndarray | List[List[int | float]], dtype: Type[BaseFloat]
) -> np.ndarray:
    """
    Converts numerical values in an array to their binary float representations
    for a given hardware number format.

    The function converts each element to its binary integer representation according
    to the specified floating point format (e.g., BF16, Float8). This is similar to
    how values would be stored in hardware registers.

    Args:
        arr: Input array or nested list of numbers to convert. Will be converted
            to numpy array if not already.
        dtype: Hardware floating point format to convert values to (e.g., BF16, Float8).
            Must be a subclass of BaseFloat.

    Returns:
        np.ndarray: Array of same shape as input but with values converted to their
            binary integer representations in the specified format.

    Example:
        >>> arr = np.array([[1.5, -2.0], [0.5, 3.25]])
        >>> binary = convert_array_dtype(arr, BF16)
        # Returns array of binary integers representing these values in BF16 format
    """
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)
    return np.vectorize(lambda x: dtype(x).binint)(arr)


def permutate_weight_matrix(arr: np.ndarray) -> np.ndarray:
    """
    Permutates a weight matrix for loading into a DiP (Diagonal Input, Permuted weight-stationary)
    systolic array configuration. Each column is cyclically shifted upward by an offset equal to
    its column index.

    This permutation ensures weights are properly aligned when they reach their target processing
    elements after propagating through the array.

    Args:
        arr: Square input matrix to be permutated

    Returns:
        np.ndarray: Permutated matrix with same shape as input

    Example:
    ```
        For a 3x3 matrix:
        Input matrix:         Permutated output:
        [a a a]              [a b c]
        [b b b]      ->      [b c a]
        [c c c]              [c a b]
    ```
        Column shifts:
        - Col 0 (a,b,c): No shift
        - Col 1 (b,c,a): Shifted up by 1 (wrapping around)
        - Col 2 (c,a,b): Shifted up by 2 (wrapping around)
    """
    rows, cols = arr.shape
    permutated = np.zeros((rows, cols))
    for i in range(cols):
        for j in range(rows):
            permutated[j][i] = arr[(j + i) % rows][i]
    return permutated


def pack_binary_vector(vec, dtype: Type[BaseFloat]):
    """Convert vector to concatenated binary representation"""
    concatenated = 0
    for i, d in enumerate(vec[::-1]):
        binary = dtype(d).binint
        concatenated += binary << (i * dtype.bitwidth())
    return concatenated


def pad_and_reshape_vector(arr, size):
    """
    Zero pads a 1D numpy array to make its length a multiple of size,
    then reshapes it to (-1, size).

    Parameters:
    -----------
    arr : numpy.ndarray
        1D input array
    size : int
        The desired width of the reshaped array

    Returns:
    --------
    numpy.ndarray
        Padded and reshaped array with shape (-1, size)
    """
    assert len(arr.shape) == 1, "Input array must be 1D"

    # Calculate the padding needed
    original_length = len(arr)
    padding_needed = (size - (original_length % size)) % size

    # Pad the array
    padded_arr = np.pad(arr, (0, padding_needed), mode="constant", constant_values=0)

    # Reshape to (-1, size)
    return padded_arr.reshape(-1, size)


def chunk_weight_matrix(matrix: np.ndarray, chunk_size: int) -> np.ndarray:
    """Splits a weight matrix into square chunks with zero padding if needed.

    Args:
        matrix: Input weight matrix to be chunked
        chunk_size: Size of each chunk (chunks will be chunk_size x chunk_size)

    Returns:
        Dictionary mapping chunk index to chunk matrix

    The function will zero-pad the input matrix if its dimensions are not
    divisible by chunk_size before splitting it into chunks.
    """
    rows, cols = matrix.shape

    # Calculate padding needed for each dimension
    pad_rows = (chunk_size - rows % chunk_size) % chunk_size
    pad_cols = (chunk_size - cols % chunk_size) % chunk_size

    # Pad matrix if needed
    padded_matrix = np.pad(matrix, ((0, pad_rows), (0, pad_cols)))

    # Calculate number of chunks in each dimension
    num_row_chunks = (rows + pad_rows) // chunk_size
    num_col_chunks = (cols + pad_cols) // chunk_size

    # Preallocate 3D array for chunks
    chunks = np.zeros((num_row_chunks * num_col_chunks, chunk_size, chunk_size))

    # Split into chunks in column-major order
    for j in range(num_col_chunks):
        for i in range(num_row_chunks):
            chunk_idx = i + j * num_row_chunks  # Column-major indexing
            r_start = i * chunk_size
            r_end = (i + 1) * chunk_size
            c_start = j * chunk_size
            c_end = (j + 1) * chunk_size
            chunks[chunk_idx] = padded_matrix[r_start:r_end, c_start:c_end]

    return chunks


def bias_trick(weights: np.ndarray, bias: np.ndarray, x: np.ndarray) -> tuple:
    """Applies bias trick to combine weights and bias into augmented matrices.

    Args:
        weights: Weight matrix (output_dim, input_dim)
        bias: Bias vector (output_dim,)
        x: Input matrix (n_samples, input_dim) or vector (input_dim,)

    Returns:
        tuple: (augmented_weights, augmented_input) where:
            - augmented_weights: (output_dim, input_dim + 1)
            - augmented_input: (n_samples, input_dim + 1) or (input_dim + 1,)
    """
    aug_weights = np.c_[weights, bias]

    # Handle both vector and matrix inputs
    if x.ndim == 1:
        aug_input = np.append(x, 1)
    else:
        aug_input = np.c_[x, np.ones(x.shape[0])]

    return aug_weights, aug_input


def count_total_gemv_tiles(layer_dims: list[tuple[int, int]], chunk_size: int) -> int:
    """Calculate total number of tiled GEMV operations for a sequence of FC layers.

    Args:
        layer_dims: List of tuples (d_in, d_out) for each layer's weight dimensions
        chunk_size: Size of tiles for GEMV operations

    Returns:
        Total number of tiled GEMV operations across all layers
    """
    total_tiles = 0

    for d_in, d_out in layer_dims:
        # Add 1 to d_in for bias term
        d_in_with_bias = d_in + 1

        # Calculate padding needed
        pad_rows = (chunk_size - d_out % chunk_size) % chunk_size
        pad_cols = (chunk_size - d_in_with_bias % chunk_size) % chunk_size

        # Calculate number of chunks in each dimension
        num_row_chunks = (d_out + pad_rows) // chunk_size
        num_col_chunks = (d_in_with_bias + pad_cols) // chunk_size

        # Each matrix-vector multiplication requires num_row_chunks * num_col_chunks tiles
        layer_tiles = num_row_chunks * num_col_chunks
        total_tiles += layer_tiles

    return total_tiles


@dataclass
class GemvTile:
    """Represents a tile for matrix-vector multiplication.

    Attributes:
        index: Index in output vector where result should be accumulated
        vector: Chunk of input vector of shape (chunk_size,)
        matrix: Chunk of matrix of shape (chunk_size, chunk_size)
        last: True if this is the last chunk for the current dest_index,
                          indicating no more accumulation needed for this output location
    """

    index: int
    """Index in output vector where result should be accumulated"""
    vector: np.ndarray
    """Chunk of input vector of shape (chunk_size,)"""
    matrix: np.ndarray
    """Chunk of matrix of shape (chunk_size, chunk_size)"""
    first: bool
    """True if this is the first chunk for the current dest_index"""
    last: bool
    """True if this is the last chunk for the current dest_index"""

    @property
    def partial_result(self) -> np.ndarray:
        """Calculate the partial result of the matrix-vector multiplication."""
        return self.vector @ self.matrix

    def __repr__(self) -> str:
        # Format arrays with numpy's array2string
        vec_str = np.array2string(self.vector, precision=4, suppress_small=True)
        mat_str = np.array2string(self.matrix, precision=4, suppress_small=True)
        result_str = np.array2string(
            self.partial_result, precision=4, suppress_small=True
        )

        return (
            f"GemvTile(\n"
            f"    index: {self.index}\n"
            f"    first: {self.first}\n"
            f"    last: {self.last}\n"
            f"    matrix:\n{mat_str}\n"
            f"    vector:\n{vec_str}\n"
            f"    partial_result:\n{result_str}\n)"
        )


def generate_gemv_tiles(
    vector: np.ndarray, matrix: np.ndarray, chunk_size: int
) -> Generator[GemvTile, None, None]:
    """Generates tiles for a matrix-vector multiplication.

    Args:
        vector: Input vector to be multiplied
        matrix: Input matrix to multiply the vector by
        chunk_size: Size of each chunk

    Yields:
        GemvTile objects containing the chunked data and metadata for processing
    """
    rows, cols = matrix.shape
    vector_len = len(vector)

    # Verify dimensions match
    assert (
        cols == vector_len
    ), f"Matrix columns must match vector length.\n{matrix.shape=} {vector.shape=}"

    # Calculate padding needed
    pad_rows = (chunk_size - rows % chunk_size) % chunk_size
    pad_cols = (chunk_size - cols % chunk_size) % chunk_size

    # Pad matrix and vector
    padded_matrix = np.pad(matrix, ((0, pad_rows), (0, pad_cols)))
    padded_vector = np.pad(vector, (0, pad_cols))

    # Calculate number of chunks
    num_row_chunks = (rows + pad_rows) // chunk_size
    num_col_chunks = (cols + pad_cols) // chunk_size

    # Generate tiles in column-major order
    for i in range(num_row_chunks):  # destination chunk index
        for j in range(num_col_chunks):  # source chunk index
            # Get the vector chunk - this stays constant for all matrix chunks
            # that contribute to the same output elements
            v_start = j * chunk_size
            v_end = (j + 1) * chunk_size
            vector_chunk = padded_vector[v_start:v_end]

            # Get the corresponding matrix chunk
            r_start = i * chunk_size
            r_end = (i + 1) * chunk_size
            c_start = j * chunk_size
            c_end = (j + 1) * chunk_size
            matrix_chunk = padded_matrix[r_start:r_end, c_start:c_end]

            # Check if this is the last column chunk for current destination
            is_last_col_chunk = j == num_col_chunks - 1
            is_first_col_chunk = j == 0

            yield GemvTile(
                index=i,
                vector=vector_chunk,
                matrix=matrix_chunk,
                last=is_last_col_chunk,
                first=is_first_col_chunk,
            )


@dataclass
class BatchGemmTile:
    """Represents a tile for batch matrix multiplication.

    Attributes:
        output_row_idx: Starting row index in output matrix where results should be accumulated
        weight_tile: Square tile from weight matrix of shape (chunk_size, chunk_size)
        batch_tile: Batch of input vectors of shape (chunk_size, batch_size)
        first: True if this is the first tile for the current output rows
        last: True if this is the last tile for the current output rows
    """

    output_row_idx: int
    weight_tile: np.ndarray
    batch_tile: np.ndarray
    first: bool
    last: bool


def generate_batch_gemm_tiles(
    weights: np.ndarray, batch_inputs: np.ndarray, chunk_size: int
) -> Generator[BatchGemmTile, None, None]:
    """Generates tiles for batch matrix multiplication optimized for systolic array processing.

    Args:
        weights: Weight matrix of shape (output_dim, input_dim)
        batch_inputs: Input batch of shape (input_dim, batch_size)
        chunk_size: Size of systolic array (NxN)

    Yields:
        BatchGemmTile objects containing chunked weights and corresponding batch inputs
    """
    out_dim, in_dim = weights.shape
    in_dim2, batch_size = batch_inputs.shape
    assert in_dim == in_dim2, "Weight and input dimensions must match"

    # Calculate padding needed for weights
    pad_out = (chunk_size - out_dim % chunk_size) % chunk_size
    pad_in = (chunk_size - in_dim % chunk_size) % chunk_size

    # Pad weights and inputs
    padded_weights = np.pad(weights, ((0, pad_out), (0, pad_in)))
    padded_inputs = np.pad(batch_inputs, ((0, pad_in), (0, 0)))

    # Calculate number of chunks
    num_out_chunks = (out_dim + pad_out) // chunk_size
    num_in_chunks = (in_dim + pad_in) // chunk_size

    # Generate tiles
    for i in range(num_out_chunks):  # Output dimension chunks
        for j in range(num_in_chunks):  # Input dimension chunks
            # Extract weight tile
            w_row_start = i * chunk_size
            w_row_end = (i + 1) * chunk_size
            w_col_start = j * chunk_size
            w_col_end = (j + 1) * chunk_size
            weight_tile = padded_weights[w_row_start:w_row_end, w_col_start:w_col_end]

            # Extract corresponding batch input tile
            batch_start = j * chunk_size
            batch_end = (j + 1) * chunk_size
            batch_tile = padded_inputs[batch_start:batch_end, :]

            is_last_input_chunk = j == num_in_chunks - 1
            is_first_input_chunk = j == 0

            yield BatchGemmTile(
                output_row_idx=i,
                weight_tile=weight_tile,
                batch_tile=batch_tile,
                first=is_first_input_chunk,
                last=is_last_input_chunk,
            )


def count_batch_gemm_tiles(output_dim: int, input_dim: int, chunk_size: int) -> int:
    """Calculate total number of tiled batch GEMM operations.

    Args:
        output_dim: Number of output dimensions (rows in weight matrix)
        input_dim: Number of input dimensions (columns in weight matrix)
        chunk_size: Size of tiles for GEMM operations (NxN)

    Returns:
        Total number of tiled GEMM operations
    """
    # Calculate padding needed
    pad_out = (chunk_size - output_dim % chunk_size) % chunk_size
    pad_in = (chunk_size - input_dim % chunk_size) % chunk_size

    # Calculate number of chunks in each dimension
    num_out_chunks = (output_dim + pad_out) // chunk_size
    num_in_chunks = (input_dim + pad_in) // chunk_size

    # Total tiles is the product of chunks in each dimension
    total_tiles = num_out_chunks * num_in_chunks

    return total_tiles


def chunk_matrices(
    matrix_a: np.ndarray, matrix_b: np.ndarray, chunk_size: int
) -> Generator[Tuple[Tuple[int, int], np.ndarray, np.ndarray], None, None]:
    """Chunks two input matrices into smaller submatrices for systolic array processing.

    Args:
        matrix_a: First input matrix of shape (M, K)
        matrix_b: Second input matrix of shape (K, N)
        chunk_size: Size N of the NxN systolic array

    Yields:
        Tuple containing:
            - (i,j) indices of the current output block
            - Chunk of matrix_a of shape (chunk_size, chunk_size)
            - Chunk of matrix_b of shape (chunk_size, chunk_size)

    The function yields chunks in the order needed to compute the final result.
    The output matrix C = A @ B will be computed block by block where each block
    is of size chunk_size x chunk_size.
    """
    M, K = matrix_a.shape
    K2, N = matrix_b.shape
    assert K == K2, "Inner dimensions must match"

    # Pad matrices if dimensions are not divisible by chunk_size
    pad_m = (chunk_size - M % chunk_size) % chunk_size
    pad_k = (chunk_size - K % chunk_size) % chunk_size
    pad_n = (chunk_size - N % chunk_size) % chunk_size

    padded_a = np.pad(matrix_a, ((0, pad_m), (0, pad_k)))
    padded_b = np.pad(matrix_b, ((0, pad_k), (0, pad_n)))

    # Number of chunks in each dimension
    m_chunks = (M + pad_m) // chunk_size
    k_chunks = (K + pad_k) // chunk_size
    n_chunks = (N + pad_n) // chunk_size

    # Iterate over output blocks
    for i in range(m_chunks):
        for j in range(n_chunks):
            # For each output block, we need to process k_chunks pairs of input blocks
            for k in range(k_chunks):
                # Extract current chunks
                a_chunk = padded_a[
                    i * chunk_size : (i + 1) * chunk_size,
                    k * chunk_size : (k + 1) * chunk_size,
                ]
                b_chunk = padded_b[
                    k * chunk_size : (k + 1) * chunk_size,
                    j * chunk_size : (j + 1) * chunk_size,
                ]

                yield (i, j), a_chunk, b_chunk


def compute_matrix_multiply(
    matrix_a: np.ndarray,
    matrix_b: np.ndarray,
    systolic_size: int,
    systolic_array_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> np.ndarray:
    """Computes matrix multiplication using chunked systolic array processing.

    Args:
        matrix_a: First input matrix of shape (M, K)
        matrix_b: Second input matrix of shape (K, N)
        systolic_size: Size N of the NxN systolic array
        systolic_array_fn: Function that takes two NxN matrices and returns their product
                          using the systolic array hardware

    Returns:
        Result matrix of shape (M, N)
    """
    M, K = matrix_a.shape
    K2, N = matrix_b.shape
    assert K == K2, "Inner dimensions must match"

    # Initialize output matrix with padding
    pad_m = (systolic_size - M % systolic_size) % systolic_size
    pad_n = (systolic_size - N % systolic_size) % systolic_size
    result = np.zeros((M + pad_m, N + pad_n))

    # Track partial sums for each output block
    block_accumulators = {}

    # Process chunks
    for (i, j), a_chunk, b_chunk in chunk_matrices(matrix_a, matrix_b, systolic_size):
        # Compute partial result using systolic array
        partial_result = systolic_array_fn(a_chunk, b_chunk)

        # Get or initialize accumulator for this output block
        if (i, j) not in block_accumulators:
            block_accumulators[(i, j)] = np.zeros((systolic_size, systolic_size))

        # Accumulate result
        block_accumulators[(i, j)] += partial_result

        # Write accumulated result to output matrix
        result[
            i * systolic_size : (i + 1) * systolic_size,
            j * systolic_size : (j + 1) * systolic_size,
        ] = block_accumulators[(i, j)]

    # Remove padding and return
    return result[:M, :N]


def simulate_systolic_multiply(chunk_a: np.ndarray, chunk_b: np.ndarray) -> np.ndarray:
    """Example systolic array simulation function - replace with actual hardware implementation.

    Args:
        chunk_a: Input matrix chunk of shape (N, N)
        chunk_b: Input matrix chunk of shape (N, N)

    Returns:
        Result of matrix multiplication of shape (N, N)
    """
    return np.dot(chunk_a, chunk_b)  # Replace with actual systolic array implementation


def visualize_matrix_chunks(
    matrix: np.ndarray, chunk_size: int, title: str = "Matrix Chunks"
):
    """Visualizes how a matrix is divided into chunks.

    Args:
        matrix: Input matrix to visualize
        chunk_size: Size of each chunk
        title: Title for the plot
    """
    M, N = matrix.shape

    # Pad matrix if needed
    pad_m = (chunk_size - M % chunk_size) % chunk_size
    pad_n = (chunk_size - N % chunk_size) % chunk_size
    padded = np.pad(matrix, ((0, pad_m), (0, pad_n)))

    # Create figure
    plt.figure(figsize=(10, 8))
    plt.imshow(padded, cmap="viridis")

    # Draw grid lines for chunks
    for i in range(0, padded.shape[0], chunk_size):
        plt.axhline(y=i - 0.5, color="red", linestyle="-", alpha=0.5)
    for j in range(0, padded.shape[1], chunk_size):
        plt.axvline(x=j - 0.5, color="red", linestyle="-", alpha=0.5)

    # Add chunk indices
    for i in range(0, padded.shape[0], chunk_size):
        for j in range(0, padded.shape[1], chunk_size):
            plt.text(
                j + chunk_size / 2 - 0.5,
                i + chunk_size / 2 - 0.5,
                f"({i//chunk_size},{j//chunk_size})",
                horizontalalignment="center",
                verticalalignment="center",
                color="white",
            )

    plt.title(f"{title}\nShape: {matrix.shape}, Chunk size: {chunk_size}x{chunk_size}")
    plt.colorbar(label="Value")
    plt.show()


def validate_matrix_multiply(
    matrix_a: np.ndarray,
    matrix_b: np.ndarray,
    systolic_size: int,
    systolic_array_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    verbose: bool = True,
) -> bool:
    """Validates the chunked matrix multiplication against numpy's implementation.

    Args:
        matrix_a: First input matrix
        matrix_b: Second input matrix
        systolic_size: Size of systolic array chunks
        systolic_array_fn: Function implementing systolic array multiplication
        verbose: Whether to print detailed information

    Returns:
        True if validation passes, False otherwise
    """
    # Compute result using our implementation
    our_result = compute_matrix_multiply(
        matrix_a, matrix_b, systolic_size, systolic_array_fn
    )

    # Compute expected result using numpy
    expected_result = np.dot(matrix_a, matrix_b)

    # Compare results
    max_diff = np.max(np.abs(our_result - expected_result))
    is_close = np.allclose(our_result, expected_result, rtol=1e-5, atol=1e-8)

    if verbose:
        print(f"Matrix A shape: {matrix_a.shape}")
        print(f"Matrix B shape: {matrix_b.shape}")
        print(f"Chunk size: {systolic_size}x{systolic_size}")
        print(f"Maximum absolute difference: {max_diff}")
        print(f"Validation {'passed' if is_close else 'failed'}")

        if not is_close:
            print("\nDetailed error analysis:")
            print(
                f"Number of elements with significant difference: "
                f"{np.sum(~np.isclose(our_result, expected_result, rtol=1e-5, atol=1e-8))}"
            )
            print("First few elements of our result:")
            print(our_result[:3, :3])
            print("\nFirst few elements of expected result:")
            print(expected_result[:3, :3])

    return is_close


def run_validation_tests():
    """Runs a series of validation tests with different matrix sizes and chunk sizes."""
    print("Running validation tests...")

    test_cases = [
        # (matrix_size, chunk_size)
        ((4, 4), 2),
        ((8, 8), 4),
        ((16, 16), 4),
        ((6, 8), 2),  # Non-square matrices
        ((15, 15), 4),  # Sizes not divisible by chunk size
    ]

    all_passed = True
    for (m, n), chunk_size in test_cases:
        print(f"\nTest case: {m}x{n} matrices with {chunk_size}x{chunk_size} chunks")

        # Create test matrices
        A = np.random.randn(m, n)
        B = np.random.randn(n, m)  # Make sure dimensions match for multiplication

        # Visualize chunking
        visualize_matrix_chunks(A, chunk_size, "Matrix A Chunks")
        visualize_matrix_chunks(B, chunk_size, "Matrix B Chunks")

        # Validate multiplication
        passed = validate_matrix_multiply(A, B, chunk_size, simulate_systolic_multiply)
        all_passed &= passed

    print("\nOverall validation:", "PASSED" if all_passed else "FAILED")
    return all_passed


if __name__ == "__main__":
    run_validation_tests()
