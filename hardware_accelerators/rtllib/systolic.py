import numpy as np
import pyrtl
from pyrtl import (
    WireVector,
    Input,
    Register,
    Simulation,
    SimulationTrace,
    reset_working_block,
)


class Buffer:
    def __init__(self, arr: np.ndarray, axis: int) -> None:
        assert axis in [0, 1], "Axis must be 0 or 1"
        if axis == 1:
            arr = arr.T
        l = arr.shape[0]
        self.arr = np.array(
            [
                np.pad(arr[i], (l - i - 1, i), "constant", constant_values=(0, 0))
                for i in range(l)
            ]
        ).T[::-1]

    def __iter__(self):
        for row in self.arr:
            yield row


class SimpleMAC:
    def __init__(self, d_width, acc_size, i, j, reset):
        self.i = i
        self.j = j
        self.data = Register(d_width, f"data_{i}_{j}")
        self.weight = Register(d_width, f"weight_{i}_{j}")
        self.acc = Register(acc_size, f"acc_{i}_{j}")
        self.reset = WireVector(1)
        self.reset <<= reset
        with pyrtl.conditional_assignment:
            with self.reset:
                self.acc.next |= 0
            with pyrtl.otherwise:
                self.acc.next |= self.acc + self.data * self.weight


class SystolicMacArray:
    def __init__(self, data: np.ndarray, weights: np.ndarray, dim=3) -> None:
        reset_working_block()
        self.size = dim
        self.cells = self._build_array(
            dim,
            data_in=pyrtl.input_list([f"data_in_{i}" for i in range(dim)], 8),
            weight_in=pyrtl.input_list([f"weight_in_{i}" for i in range(dim)], 8),
        )
        self.data = Buffer(data, axis=0)
        self.weights = Buffer(weights, axis=1)
        self.sim = pyrtl.Simulation(tracer=SimulationTrace())

    def _build_array(
        self, size: int, data_in: list[WireVector], weight_in: list[WireVector]
    ):
        # Create the mac units
        reset_wire = Input(1, "rst")
        cells = [
            [SimpleMAC(8, 16, i, j, reset_wire) for j in range(size)]
            for i in range(size)
        ]
        # Connect them together
        for row in range(size):
            for col in range(size):
                # Connect data inputs
                if col == 0:  # First column gets data from the data_in buffer
                    cells[row][col].data.next <<= data_in[row]
                else:  # Other columns get data from the "left" neighbor
                    cells[row][col].data.next <<= cells[row][col - 1].data
                # Connect weight inputs
                if row == 0:  # Top row gets weights from the weight_in buffer
                    cells[row][col].weight.next <<= weight_in[col]
                else:  # Other rows get weights from the top neighbor
                    cells[row][col].weight.next <<= cells[row - 1][col].weight
        return cells

    def __iter__(self):
        for d, w in zip(self.data, self.weights):
            sim_inputs = {
                **{f"data_in_{i}": v for i, v in enumerate(d)},
                **{f"weight_in_{i}": v for i, v in enumerate(w)},
            }
            print(f"data={d}, weights={w}")
            self.sim.step(sim_inputs)
            yield self.get_current_state()
        for _ in range(self.size + 1):
            sim_inputs = {
                **{f"data_in_{i}": 0 for i in range(self.size)},
                **{f"weight_in_{i}": 0 for i in range(self.size)},
            }
            self.sim.step(sim_inputs)
            yield self.get_current_state()

    def get_current_state(self):
        cur = []
        for row in self.cells:
            cur.append([f"{self.sim.inspect(mac.acc.name)}" for mac in row])
        return np.array(cur)

    def matmul(self, data: np.ndarray, weights: np.ndarray, v=True):
        z = np.zeros((self.size, self.size + 1))
        data_buffer = np.concat([Buffer(data, axis=0).arr.T, z], axis=1)
        weight_buffer = np.concat([Buffer(weights, axis=1).arr.T, z], axis=1)
        sim_inputs = {
            **{f"data_in_{i}": v for i, v in enumerate(data_buffer)},
            **{f"weight_in_{i}": v for i, v in enumerate(weight_buffer)},
            "rst": [0, 0, 0, 0, 0, 0, 0, 0, 0],
        }
        print(sim_inputs)
        sim = Simulation()
        sim.step_multiple(sim_inputs, nsteps=4)
        result = self.get_current_state()
        print("Systolic Result:", result)
        print("True Result:", data @ weights)
        return result
