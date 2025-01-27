import numpy as np
import pyrtl
from typing import Callable, Dict, List, Type
from pyrtl.simulation import default_renderer
from typing import Callable, List
import subprocess
import os
import sys
import datetime
from pyrtl.simulation import default_renderer

from ..dtypes.base import BaseFloat


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
        from IPython.display import display, HTML, Javascript  # type: ignore

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
