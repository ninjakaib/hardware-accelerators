import numpy as np
from typing import List, Tuple, Generator, Callable
import matplotlib.pyplot as plt

def chunk_matrices(matrix_a: np.ndarray, matrix_b: np.ndarray, chunk_size: int) -> Generator[Tuple[Tuple[int, int], np.ndarray, np.ndarray], None, None]:
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
                    i * chunk_size:(i + 1) * chunk_size,
                    k * chunk_size:(k + 1) * chunk_size
                ]
                b_chunk = padded_b[
                    k * chunk_size:(k + 1) * chunk_size,
                    j * chunk_size:(j + 1) * chunk_size
                ]
                
                yield (i, j), a_chunk, b_chunk

def compute_matrix_multiply(
    matrix_a: np.ndarray, 
    matrix_b: np.ndarray, 
    systolic_size: int,
    systolic_array_fn: Callable[[np.ndarray, np.ndarray], np.ndarray]
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
            i * systolic_size:(i + 1) * systolic_size,
            j * systolic_size:(j + 1) * systolic_size
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

def visualize_matrix_chunks(matrix: np.ndarray, chunk_size: int, title: str = "Matrix Chunks"):
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
    plt.imshow(padded, cmap='viridis')
    
    # Draw grid lines for chunks
    for i in range(0, padded.shape[0], chunk_size):
        plt.axhline(y=i-0.5, color='red', linestyle='-', alpha=0.5)
    for j in range(0, padded.shape[1], chunk_size):
        plt.axvline(x=j-0.5, color='red', linestyle='-', alpha=0.5)
    
    # Add chunk indices
    for i in range(0, padded.shape[0], chunk_size):
        for j in range(0, padded.shape[1], chunk_size):
            plt.text(j + chunk_size/2 - 0.5, i + chunk_size/2 - 0.5, 
                    f'({i//chunk_size},{j//chunk_size})',
                    horizontalalignment='center',
                    verticalalignment='center',
                    color='white')
    
    plt.title(f"{title}\nShape: {matrix.shape}, Chunk size: {chunk_size}x{chunk_size}")
    plt.colorbar(label='Value')
    plt.show()

def validate_matrix_multiply(
    matrix_a: np.ndarray,
    matrix_b: np.ndarray,
    systolic_size: int,
    systolic_array_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    verbose: bool = True
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
    our_result = compute_matrix_multiply(matrix_a, matrix_b, systolic_size, systolic_array_fn)
    
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
            print(f"Number of elements with significant difference: "
                  f"{np.sum(~np.isclose(our_result, expected_result, rtol=1e-5, atol=1e-8))}")
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