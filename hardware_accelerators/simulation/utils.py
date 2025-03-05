import datetime
import math
import os
import subprocess
import sys
from typing import Callable, Dict, List, Literal, Type

import numpy as np
import pyrtl
from pyrtl.simulation import default_renderer

from ..dtypes.base import BaseFloat


def render_waveform(
    sim: pyrtl.Simulation,
    trace_list: List[str] | None = None,
    repr_func: Callable[[int], str] = str,
    repr_per_name: Dict[str, Callable[[int], str]] = {},
    output_file: str | None = None,
):
    # Generate the HTML string
    htmlstring = pyrtl.trace_to_html(
        sim.tracer,
        trace_list=trace_list,
        repr_func=repr_func,
        repr_per_name=repr_per_name,
    )

    # Create complete HTML document with required WaveDrom scripts
    full_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/wavedrom/1.6.2/skins/default.js"></script>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/wavedrom/1.6.2/wavedrom.min.js"></script>
    </head>
    <body>
        {}
        <script>
            window.onload = function() {{
                WaveDrom.ProcessAll();
            }}
        </script>
    </body>
    </html>
    """.format(
        htmlstring
    )

    if output_file:
        # Write to file if output_file is specified
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(full_html)
        print(f"Trace written to {output_file}")
    else:
        # Original display behavior for notebook environment
        from IPython.display import HTML, Javascript, display  # type: ignore

        html_elem = HTML(htmlstring)
        display(html_elem)
        js_stuff = """
        $.when(
        $.getScript("https://cdnjs.cloudflare.com/ajax/libs/wavedrom/1.6.2/skins/default.js"),
        $.getScript("https://cdnjs.cloudflare.com/ajax/libs/wavedrom/1.6.2/wavedrom.min.js"),
        $.Deferred(function( deferred ){
            $( deferred.resolve );
        })).done(function(){
            WaveDrom.ProcessAll();
        });"""
        display(Javascript(js_stuff))


def calculate_accumulator_memory(
    accum_addr_width: int,
    array_size: int,
    dtype: Type[BaseFloat],
    unit: Literal["B", "KB", "MB"] = "KB",
    verbose: bool = False,
) -> float:
    conversions = {
        "B": 8.0,
        "KB": 8.0 * 1024,
        "MB": 8.0 * 1024 * 1024,
    }
    bits = array_size * (2**accum_addr_width) * dtype.bitwidth()
    mem = bits / conversions[unit]
    slots = 2**accum_addr_width
    if verbose:
        print(
            f"{mem} {unit} ({slots} slots) avaialable for {accum_addr_width} address bits",
            f"with {array_size}x{array_size} array in {dtype.__name__}",
        )
    return mem


def calculate_accum_addr_width_for_min_mem(
    required_mem: float,
    array_size: int,
    dtype: Type[BaseFloat],
    unit: Literal["B", "KB", "MB"] = "KB",
    verbose: bool = False,
) -> int:
    conversions = {
        "B": 8.0,
        "KB": 8.0 * 1024,
        "MB": 8.0 * 1024 * 1024,
    }
    bits = required_mem * conversions[unit]
    req_width = math.ceil(math.log2(bits / (array_size * dtype.bitwidth())))
    slots = 2**req_width
    if verbose:
        print(
            f"{req_width} address bits ({slots} slots) required for >= {required_mem} {unit}",
            f"with {array_size}x{array_size} array in {dtype.__name__}",
        )
    return req_width


def calculate_accum_addr_width_for_min_slots(
    slots: int,
    verbose: bool = False,
) -> int:
    req_width = math.ceil(math.log2(slots))
    available = 2**req_width
    if verbose:
        print(
            f"{req_width} address bits required for >= {slots} slots",
            f"({available} slots available)",
        )
    return req_width
