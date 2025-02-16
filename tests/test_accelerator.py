from hardware_accelerators import BF16, float_adder, lmul_fast, float_multiplier
from hardware_accelerators.rtllib import TiledAcceleratorConfig
from hardware_accelerators.simulation import TiledMatrixEngineSimulator
import numpy as np

from hardware_accelerators.simulation.accelerator import AcceleratorSimulator


def test_accelerator_basic():
    simulator = AcceleratorSimulator.default_config(array_size=3, num_weight_tiles=2)

    simulator.setup()

    weights = np.ones((3, 3))
    activations = np.array([[1, 2, 3], [-4, -5, -6], [7, 8, 9]])

    simulator.load_weights(weights, 0)

    simulator.execute_instruction(
        data_vec=activations[0],
        load_new_weights=True,
        flush_pipeline=False,
        activation_enable=True,
        activation_func="relu",
    )
    simulator.execute_instruction(
        data_vec=activations[1],
        accum_addr=1,
        flush_pipeline=False,
        activation_enable=True,
        activation_func="relu",
    )
    simulator.execute_instruction(
        data_vec=activations[2],
        accum_addr=2,
        activation_enable=True,
        activation_func="relu",
        flush_pipeline=True,
    )

    results = np.zeros((activations.shape[0], weights.shape[1]))

    for i in range(3):
        results[i] = simulator._get_outputs()
        simulator.execute_instruction(nop=True)

    gt = np.maximum(0, (activations @ weights))

    assert np.isclose(results, gt).all()


def test_matrix_engine_matmul():
    """Test the matmul function."""
    config = TiledAcceleratorConfig(
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

    sim = TiledMatrixEngineSimulator(config)

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
