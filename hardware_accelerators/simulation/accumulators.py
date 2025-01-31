from dataclasses import dataclass
from typing import Any, List
from ..rtllib.accumulators import TiledAccumulatorFSM, TiledAddressGenerator
from pyrtl import Input, Output, Simulation, reset_working_block


@dataclass
class TiledAddressGeneratorExpectedState:
    """Expected state at each simulation step"""

    state: TiledAccumulatorFSM
    row: int
    addr: int
    write_enable: bool


@dataclass
class TiledAddressGeneratorState:
    """Stores the state of the address generator at a given simulation step"""

    inputs: dict[str, Any]  # tile_addr, start, write_valid
    state: TiledAccumulatorFSM
    current_row: int
    write_addr: int
    write_enable: int
    step: int

    def __repr__(self) -> str:
        """Pretty print the simulation state"""
        width = 40
        sep = "-" * width
        return (
            f"\nAddress Generator State - Step {self.step}\n{sep}\n"
            f"Inputs:\n"
            f"  tile_addr: {self.inputs['tile_addr']}\n"
            f"  start: {self.inputs['start']}\n"
            f"  write_valid: {self.inputs['write_valid']}\n"
            f"FSM State: {self.state.name}\n"
            f"Current Row: {self.current_row}\n"
            f"Write Address: {self.write_addr}\n"
            f"Write Enable: {self.write_enable}\n"
            f"{sep}\n"
        )


class TiledAddressGeneratorSimulator:
    def __init__(
        self,
        array_size: int,
        num_tiles: int,
    ):
        """Initialize address generator simulator

        Args:
            array_size: Dimension of systolic array (NxN)
            num_tiles: Number of tiles to support
        """
        self.array_size = array_size
        self.num_tiles = num_tiles
        self.tile_addr_width = (num_tiles - 1).bit_length()
        self.history: List[TiledAddressGeneratorState] = []

    def _setup(self):
        """Setup PyRTL simulation environment"""
        reset_working_block()

        # Create inputs
        self.tile_addr = Input(self.tile_addr_width, "tile_addr")
        self.start = Input(1, "start")
        self.write_valid = Input(1, "write_valid")

        # Create address generator
        self.addr_gen = TiledAddressGenerator(
            tile_addr_width=self.tile_addr_width, array_size=self.array_size
        )

        # Connect signals
        self.addr_gen.connect_tile_addr(self.tile_addr)
        self.addr_gen.connect_start(self.start)
        self.addr_gen.connect_write_valid(self.write_valid)

        # Create simulation
        self.sim = Simulation()
        self.sim_inputs = {"tile_addr": 0, "start": 0, "write_valid": 0}

    def simulate(
        self, steps: List[dict], verbose: bool = False
    ) -> List[TiledAddressGeneratorState]:
        """Run simulation with provided input sequence

        Args:
            steps: List of dictionaries containing input values for each step
            verbose: If True, print detailed cycle-by-cycle analysis
        """
        self._setup()
        self.history = []

        if verbose:
            print("\nCycle by cycle analysis:")
            print("-" * 70)
            print("Cycle | Inputs      | State    | Row | Addr | WE | Notes")
            print("-" * 70)

        for i, step in enumerate(steps):
            self.sim_inputs.update(step)
            self._step()

            if verbose:
                state = self.history[-1]
                # Generate note about what's happening
                if step["start"] and state.state == TiledAccumulatorFSM.IDLE:
                    note = "Starting new tile"
                elif state.state == TiledAccumulatorFSM.WRITING and step["write_valid"]:
                    note = f"Writing to tile {step['tile_addr']}"
                elif step["start"] and state.state == TiledAccumulatorFSM.WRITING:
                    note = "Start ignored (busy)"
                else:
                    note = ""

                print(
                    f"{i:5d} | "
                    f"t={step['tile_addr']} "
                    f"s={step['start']} "
                    f"v={step['write_valid']} | "
                    f"{state.state.name:8s} | "
                    f"{state.current_row:3d} | "
                    f"{state.write_addr:4d} | "
                    f"{state.write_enable:2d} | "
                    f"{note}"
                )

        return self.history

    def _step(self):
        """Advance simulation one step and record state"""
        self.sim.step(self.sim_inputs)

        # Record simulation state
        state = TiledAddressGeneratorState(
            inputs=self.sim_inputs.copy(),
            state=TiledAccumulatorFSM(self.sim.inspect(self.addr_gen.state.name)),
            current_row=self.sim.inspect(self.addr_gen.current_row.name),
            write_addr=self.sim.inspect(self.addr_gen.internal_write_addr.name),
            write_enable=self.sim.inspect(self.addr_gen.internal_write_enable.name),
            step=len(self.history),
        )
        self.history.append(state)

    def compute_expected_behavior(
        self, steps: List[dict]
    ) -> List[TiledAddressGeneratorExpectedState]:
        """Compute expected behavior without using simulation

        Models the expected FSM behavior and address generation in software,
        completely independent of the hardware implementation.

        Args:
            steps: List of input dictionaries

        Returns:
            List of expected states for each cycle
        """
        expected_history = []
        state = TiledAccumulatorFSM.IDLE
        row = 0
        addr = 0

        # Compute base addresses for all tiles
        tile_bases = [self.array_size * i for i in range(2**self.tile_addr_width)]

        for i, step in enumerate(steps):
            # Calculate write enable
            write_enable = state == TiledAccumulatorFSM.WRITING and step["write_valid"]

            # Store current expected state
            expected_history.append(
                TiledAddressGeneratorExpectedState(
                    state=state, row=row, addr=addr, write_enable=write_enable
                )
            )

            # Calculate next state and outputs
            if state == TiledAccumulatorFSM.IDLE:
                if step["start"]:
                    state = TiledAccumulatorFSM.WRITING
                    addr = tile_bases[step["tile_addr"]]
                    row = 0

            elif state == TiledAccumulatorFSM.WRITING:
                if step["write_valid"]:
                    if row == self.array_size - 1:
                        state = TiledAccumulatorFSM.IDLE
                        row = 0
                    else:
                        row += 1
                        addr += 1

        return expected_history
