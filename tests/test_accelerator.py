from hardware_accelerators import BF16, float_adder, lmul_fast, float_multiplier
from hardware_accelerators.rtllib import AcceleratorConfig
from hardware_accelerators.simulation import MatrixEngineSimulator
import numpy as np


def test_accelerator_matmul():
    """Test the matmul function."""
    config = AcceleratorConfig(
        array_size=3,
        data_type=BF16,
        weight_type=BF16,
        accum_type=BF16,
        pe_adder=float_adder,
        pe_multiplier=float_multiplier,
        accum_adder=float_adder,
        pipeline=False,
        accumulator_tiles=4,
    )

    sim = MatrixEngineSimulator(config)

    weights = np.identity(3)
    activations = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    # Load data
    print(f"Original Weights:\n{weights}")
    print(f"Original Activations:\n{activations}\n")

    sim.load_weights(weights, bank=0)
    sim.load_activations(activations, bank=0)

    # Perform computation
    sim.matmul(data_bank=0, weight_bank=0, accum_tile=0, accumulate=False)
    # sim.matmul(data_bank=0, weight_bank=0, accum_tile=0, accumulate=True)

    # Debug
    # sim.print_history(inputs=True, memory_buffer=False, accumulator=True)
    tile = sim.read_accumulator_tile(0)

    assert np.isclose(tile[::-1], activations @ weights).all()
