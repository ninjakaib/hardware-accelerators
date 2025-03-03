import os
from pathlib import Path
import pickle
import ctypes
import _ctypes
import platform
import shutil
from typing import Literal, Optional, Dict, List, Union, Any

import numpy as np
from pyrtl import (
    CompiledSimulation,
    SimulationTrace,
    PyrtlError,
    working_block,
    Block,
    Input,
    Output,
)

from ..nn.util import softmax

from ..nn.mlp import MLP

from .matrix_utils import (
    bias_trick,
    count_batch_gemm_tiles,
    generate_batch_gemm_tiles,
    generate_gemv_tiles,
    permutate_weight_matrix,
)

from ..rtllib.accelerator import (
    CompiledAccelerator,
    CompiledAcceleratorConfig,
)


class ReusableCompiledSimulation(CompiledSimulation):
    """Extension of CompiledSimulation that supports loading from precompiled libraries."""

    def __init__(self, lib_path: Optional[str] = None):
        """Initialize either from scratch or from a precompiled library.

        Args:
            lib_path: Path to precompiled library. If provided, loads from binary.
                     If None, initializes normally (compiles a new binary).
        """
        if lib_path is None:
            # Normal initialization - will compile the binary
            super().__init__()  # type: ignore
        else:
            # Load from precompiled library - skip compilation
            self._load_from_binary(lib_path)

    def _load_from_binary(self, lib_path: str | Path):
        """Load simulation from a precompiled binary and state file.

        Args:
            lib_path: Path to either the .so file or the directory containing it.
        """
        lib_path = Path(lib_path)

        # Check if lib_path is a directory or a file
        if lib_path.is_dir():
            # It's a directory, look for pyrtlsim.so inside it
            so_path = lib_path / "pyrtlsim.so"
            state_path = lib_path / "state.pkl"
        else:
            # It's a file path, assume it's the .so file
            so_path = lib_path
            # Check if state file is in the same directory with generic name
            state_path = lib_path.parent / "state.pkl"
            if not state_path.exists():
                # Try the old naming convention
                state_path = lib_path.with_suffix(".state")

        # Verify files exist
        if not so_path.exists():
            raise PyrtlError(f"Library file not found: {so_path}")
        if not state_path.exists():
            raise PyrtlError(f"State file not found: {state_path}")

        # Load the saved state
        with open(state_path, "rb") as f:
            state = pickle.load(f)

        # Restore state
        self._dll = None
        self._dir = None
        self.block = state["block"]
        self.tracer = state["tracer"]
        self.default_value = state["default_value"]
        self._regmap = state["regmap"]
        self._memmap = state["memmap"]
        self._uid_counter = state["uid_counter"]
        self.varname = state["varname"]
        self._inputpos = state["inputpos"]
        self._outputpos = state["outputpos"]
        self._inputbw = state["inputbw"]
        self._ibufsz = state["ibufsz"]
        self._obufsz = state["obufsz"]
        self._probe_mapping = state.get("probe_mapping", {})

        # Load the DLL
        self._dll = ctypes.CDLL(lib_path)
        self._crun = self._dll.sim_run_all
        self._crun.restype = None
        self._initialize_mems = self._dll.initialize_mems
        self._initialize_mems.restype = None
        self._mem_lookup = self._dll.lookup
        self._mem_lookup.restype = ctypes.POINTER(ctypes.c_uint64)

        # Initialize memories
        self._initialize_mems()

    def save_compiled_lib(self, save_dir: Path):
        """Save the compiled library and state to a permanent location.

        Args:
            save_dir: Base directory to save the files to.

        Returns:
            Path to the directory where files were saved
        """
        save_dir.mkdir(parents=True, exist_ok=True)

        if self._dll is None or self._dir is None:
            raise PyrtlError("No compiled library exists to save")

        # Copy library
        shutil.copy2(Path(self._dir) / "pyrtlsim.so", save_dir / "pyrtlsim.so")

        # Save state
        state = {
            "block": self.block,
            "tracer": self.tracer,
            "default_value": self.default_value,
            "regmap": self._regmap,
            "memmap": self._memmap,
            "uid_counter": self._uid_counter,
            "varname": self.varname,
            "inputpos": self._inputpos,
            "outputpos": self._outputpos,
            "inputbw": self._inputbw,
            "ibufsz": self._ibufsz,
            "obufsz": self._obufsz,
            "probe_mapping": getattr(self, "_probe_mapping", {}),
        }

        with open(save_dir / "state.pkl", "wb") as f:
            pickle.dump(state, f)

        return save_dir
