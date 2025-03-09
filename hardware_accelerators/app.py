import os
from typing import Literal
import gradio as gr
from gradio.components.image_editor import EditorValue
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
from .nn.util import load_model
from .rtllib.lmul import lmul_simple
from .rtllib.multipliers import float_multiplier
from .dtypes import Float8, BF16
from .rtllib import (
    CompiledAcceleratorConfig,
)
from .simulation import CompiledAcceleratorSimulator
from .analysis.hardware_stats import (
    calculate_hardware_stats,
)

__all__ = ["create_app"]

# ------------ CONSTANTS ------------ #

# Load the component data
data_path = os.environ.get("COMPONENT_DATA_PATH", "results/component_data.csv")
DF = pd.read_csv(data_path)

# Load the trained model
MODEL = load_model("models/mlp_mnist.pth", "cpu")  # type: ignore
MODEL.eval()

classes = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
]
labels_value = {label: 0.0 for label in classes}

accelerator_dtypes = ["float8", "bfloat16", "float32"]
dtype_map = {
    "float8": Float8,
    "bfloat16": BF16,
    "float32": BF16,  # TODO: use Float32, but not right now because its slow
}


mult_map = {
    "IEEE 754": float_multiplier,
    "l-mul": lmul_simple,
}


# ------------ Event Listener Functions ------------ #


def filter_activation_types(weight_type: str, activation_type: str):
    if weight_type == "float8":
        return gr.update(choices=accelerator_dtypes)
    elif weight_type == "bfloat16":
        if activation_type == "float8":
            activation_type = "bfloat16"
        return gr.update(value=activation_type, choices=["bfloat16", "float32"])
    elif weight_type == "float32":
        if activation_type != "float32":
            activation_type = "float32"
        return gr.update(value=activation_type, choices=["float32"])


def warn_w8a8(weight_type: str, activation_type: str):
    if weight_type == "float8" and activation_type == "float8":
        gr.Warning(
            "W8A8 has poor performance without quantization, which is not yet supported in simulation. Theoretical results are still calculated for FP8 hardware",
            duration=5,
        )


def image_to_tensor(sketchpad: EditorValue):
    image = sketchpad["composite"]
    image = image.resize((28, 28), Image.Resampling.LANCZOS)  # type: ignore
    img_array = np.transpose(np.array(image), (2, 0, 1))[-1]

    # Preprocessing: convert image to tensor and normalize
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    tensor_image = transform(img_array)
    return tensor_image


def calculate_stats(
    activation_type: Literal["float8", "bfloat16", "float32"],
    weight_type: Literal["float8", "bfloat16", "float32"],
    systolic_array_size: int,
    num_accelerator_cores: int,
    fast_internals: Literal["Fast", "Efficient"],
    pipeline_level: Literal["None", "Low", "Full"],
    process_node_size: Literal["7nm", "45nm", "130nm"],
):
    """
    Calculate hardware statistics for both lmul and standard IEEE multiplier configurations.

    Args:
        activation_type: Type of activations
        weight_type: Type of weights
        systolic_array_size: Size of the systolic array
        num_accelerator_cores: Number of accelerator cores
        fast_internals: Whether to use fast or efficient components
        pipeline_level: Level of pipelining
        process_node_size: Process node size (ignored for now)

    Returns:
        Tuple of (lmul_metrics, ieee_metrics, comparison_metrics) dictionaries
    """
    stat_map = {
        "float8": "fp8",
        "bfloat16": "bf16",
        "float32": "fp32",
        "Fast": True,
        "Efficient": False,
        "None": 0,
        "Low": 1,
        "None": "combinational",
        "Low": "combinational",
        "Full": "pipelined",
        "7nm": 7,
        "45nm": 45,
        "130nm": 130,
    }
    # Calculate hardware stats using the functions from hardware_stats.py
    lmul_metrics, ieee_metrics = calculate_hardware_stats(
        DF,
        activation_type,
        weight_type,
        systolic_array_size,
        num_accelerator_cores,
        fast_internals,
        pipeline_level,
        process_node_size,
    )

    # comparison_metrics = calculate_comparison_metrics(lmul_metrics, ieee_metrics)

    # Format the metrics for display in the Gradio UI
    lmul_html = "<div style='text-align: left;'>"
    for key, value in lmul_metrics.items():
        lmul_html += f"<p><b>{key}:</b> {value}</p>"
    lmul_html += "</div>"

    ieee_html = "<div style='text-align: left;'>"
    for key, value in ieee_metrics.items():
        ieee_html += f"<p><b>{key}:</b> {value}</p>"
    ieee_html += "</div>"

    # comparison_html = "<div style='text-align: left;'>"
    # comparison_html += "<h3>Comparison (lmul vs IEEE)</h3>"
    # for key, value in comparison_metrics.items():
    #     comparison_html += f"<p><b>{key}:</b> {value}</p>"
    # comparison_html += "</div>"

    return (
        lmul_html,
        ieee_html,
        # comparison_html,
    )


def predict_lmul(
    sketchpad: EditorValue,
    weight: str,
    activation: str,
    gr_progress=gr.Progress(track_tqdm=True),
):
    if weight == "float8" and activation == "float8":
        activation = "bfloat16"
    config = CompiledAcceleratorConfig(
        array_size=8,
        activation_type=dtype_map[activation],
        weight_type=dtype_map[weight],
        multiplier=lmul_simple,
    )
    sim = CompiledAcceleratorSimulator(config, MODEL)

    x = image_to_tensor(sketchpad).detach().numpy().flatten()
    probabilities = sim.predict(x)
    return {cls: float(prob) for cls, prob in zip(classes, probabilities)}


def predict_ieee(
    sketchpad: EditorValue,
    weight: str,
    activation: str,
    gr_progress=gr.Progress(track_tqdm=True),
):
    if weight == "float8" and activation == "float8":
        activation = "bfloat16"
    config = CompiledAcceleratorConfig(
        array_size=8,
        activation_type=dtype_map[activation],
        weight_type=dtype_map[weight],
        multiplier=float_multiplier,
    )
    simulator = CompiledAcceleratorSimulator(config, MODEL)

    x = image_to_tensor(sketchpad).detach().numpy().flatten()
    probabilities = simulator.predict(x)
    return {cls: float(prob) for cls, prob in zip(classes, probabilities)}


# ------------ Blocks UI Layout ------------ #


def create_app():
    with gr.Blocks(fill_height=False, fill_width=False, title=__file__) as demo:

        gr.Markdown("## Draw a digit to see the model's prediction")
        with gr.Row(equal_height=False):
            with gr.Column(scale=3):
                canvas_size = (400, 400)
                sketchpad = gr.Sketchpad(
                    # label="Draw a digit",
                    type="pil",  # Changed to PIL
                    transforms=(),
                    layers=False,
                    canvas_size=canvas_size,
                    # scale=2,
                    container=False,
                )
                predict_btn = gr.Button(
                    "Run Hardware Simulation",
                    variant="primary",
                )

                # with gr.Accordion("Accelerator Configuration", open=True):
                with gr.Group():
                    with gr.Row():  # Weight and activation types
                        weight_type_component = gr.Radio(
                            label="Weights d-type",
                            choices=accelerator_dtypes,
                            value="float8",
                            interactive=True,
                        )
                        activation_type_component = gr.Radio(
                            label="Activations d-type",
                            choices=accelerator_dtypes,
                            value="bfloat16",
                            interactive=True,
                        )
                        # Prevent w8a8 from being selected, or any other combination where act < weight
                        weight_type_component.select(
                            fn=filter_activation_types,
                            inputs=[weight_type_component, activation_type_component],
                            outputs=activation_type_component,
                        )
                        gr.on(
                            triggers=[
                                weight_type_component.select,
                                activation_type_component.select,
                            ],
                            fn=warn_w8a8,
                            inputs=[weight_type_component, activation_type_component],
                        )
                    with gr.Row():
                        systolic_array_size_component = gr.Slider(
                            label="Systolic Array Size",
                            info="Dimensions of the matrix acceleration unit",
                            minimum=4,
                            maximum=512,
                            step=1,
                            value=16,
                            interactive=True,
                        )
                        num_accelerator_cores_component = gr.Number(
                            label="Number of Accelerator Cores",
                            info="Total number of accelerator units per chip",
                            minimum=1,
                            maximum=1024,
                            step=1,
                            value=1,
                            interactive=True,
                        )
                    with gr.Row(equal_height=True):
                        fast_internals_component = gr.Dropdown(
                            label="Internal Component Type",
                            info="Configure the lowest level hardware units to use a faster or more efficient design.",
                            choices=["Fast", "Efficient"],
                            value="Fast",
                            interactive=True,
                            filterable=False,
                        )
                        pipeline_level_component = gr.Dropdown(
                            label="Pipeline Level",
                            info="Configure the pipeline level of processing elements within the accelerator. Low uses a single register between multipliers and adders. Full uses pipelined individual components.",
                            choices=["None", "Low", "Full"],
                            value="Full",
                            interactive=True,
                            filterable=False,
                        )
                        process_node_size_component = gr.Radio(
                            label="Process Node Size",
                            info="Configure the process node size of the hardware units. Smaller nodes are faster and use less area.",
                            choices=["7nm", "45nm", "130nm"],
                            value="45nm",
                            interactive=False,
                        )

            with gr.Column(scale=2):
                lmul_predictions = gr.Label(
                    label="l-mul Simulator Predictions",
                    value=labels_value,
                    min_width=100,
                )

                lmul_html = gr.HTML(
                    label="L-mul Hardware Stats",
                    show_label=True,
                    container=True,
                )

            with gr.Column(scale=2):
                ieee_predictions = gr.Label(
                    label="Hardware Simulator Predictions",
                    value=labels_value,
                    min_width=100,
                )
                ieee_html = gr.HTML(
                    label="IEEE Multiplier Hardware Stats",
                    show_label=True,
                    container=True,
                )

        # ------------ Event Listeners ------------ #

        predict_btn.click(
            fn=predict_ieee,
            inputs=[sketchpad, weight_type_component, activation_type_component],
            outputs=ieee_predictions,
        )

        # TODO: implement simulator_predict
        predict_btn.click(
            fn=predict_lmul,
            inputs=[sketchpad, weight_type_component, activation_type_component],
            outputs=lmul_predictions,
        )

        gr.on(
            triggers=[
                demo.load,
                activation_type_component.change,
                weight_type_component.change,
                systolic_array_size_component.change,
                num_accelerator_cores_component.change,
                fast_internals_component.change,
                pipeline_level_component.change,
                process_node_size_component.change,
            ],
            fn=calculate_stats,
            inputs=[
                activation_type_component,
                weight_type_component,
                systolic_array_size_component,
                num_accelerator_cores_component,
                fast_internals_component,
                pipeline_level_component,
                process_node_size_component,
            ],
            outputs=[lmul_html, ieee_html],
            show_progress="hidden",
        )

    return demo


if __name__ == "__main__":
    demo = create_app()
    demo.queue()
    demo.launch(share=False)
