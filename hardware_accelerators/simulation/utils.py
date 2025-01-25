import pyrtl
from typing import Callable, List
from pyrtl.simulation import default_renderer
from typing import Callable, List
import subprocess
import os
import sys
import datetime
from pyrtl.simulation import default_renderer


def render_waveform(
    sim: pyrtl.Simulation,
    trace_list: List[str] = None,
    output_file: str = None,
    renderer=default_renderer(),
    symbol_len: int = None,
    repr_func: Callable[[int], str] = hex,
    repr_per_name={},
    segment_size: int = 1,
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
        from IPython.display import display, HTML, Javascript

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
