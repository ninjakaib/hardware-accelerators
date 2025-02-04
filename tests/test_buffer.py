from hardware_accelerators.simulation.buffer import BufferMemorySimulator
import numpy as np


def test_buffer_memory_basic():
    # Test parameters
    SIZE = 3

    # Create and setup simulator
    sim = BufferMemorySimulator(array_size=SIZE).setup()

    # Test data
    data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])

    weights = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])

    # Load memories
    print("\nLoading memories...")
    sim.load_memories(data_bank=0, weight_bank=0, data=data, weights=weights)

    # Stream data
    print("\nStreaming data...")
    data_vectors = sim.stream_data(data_bank=0)

    print("\nData vectors streamed:")
    for i, vec in enumerate(data_vectors):
        print(f"Step {i}: {np.array2string(np.array(vec), precision=4)}")

    # Stream weights
    print("\nStreaming weights...")
    weight_vectors = sim.stream_weights(weight_bank=0)

    print("\nWeight vectors streamed:")
    for i, vec in enumerate(weight_vectors):
        print(f"Step {i}: {np.array2string(np.array(vec), precision=4)}")

    # Verify correct streaming order
    expected_data = data  # For this simple case, expect rows in order
    expected_weights = weights  # For this simple case, expect rows in order

    actual_data = np.array(data_vectors)
    actual_weights = np.array(weight_vectors)

    print("\nVerifying results...")
    np.testing.assert_allclose(actual_data, expected_data, rtol=0.01)
    np.testing.assert_allclose(actual_weights, expected_weights, rtol=0.01)
    print("All tests passed!")
