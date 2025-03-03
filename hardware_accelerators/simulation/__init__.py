import os
from pathlib import Path
from platformdirs import user_cache_dir

from .repr_funcs import *
from .matrix_utils import *
from .utils import render_waveform
from .systolic import SystolicArraySimulator
from .accumulators import AccumulatorBankSimulator
from .buffer import BufferMemorySimulator
from .accelerator import (
    CompiledAcceleratorSimulator,
    AcceleratorSimulator,
    TiledMatrixEngineSimulator,
)


def get_sim_cache_dir():
    # Check both system env and .env file
    if env_value := os.getenv("HWA_SIM_CACHE"):
        return Path(env_value).expanduser().resolve()

    return Path(user_cache_dir("hardware_accelerators", ensure_exists=True))


def set_sim_cache_dir(path: Path):
    os.environ["HWA_SIM_CACHE"] = str(path)
    return get_sim_cache_dir()
