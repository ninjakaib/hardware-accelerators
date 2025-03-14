import os
from typing import Literal
import gradio as gr
from gradio.components.image_editor import EditorValue
import numpy as np
import pandas as pd
from PIL import Image
import struct
import random
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


# ------------ MNIST Dataset Loading ------------ #


def load_mnist_images(images_path, labels_path, num_images=100):
    """
    Load MNIST images and labels from the IDX format files.

    Args:
        images_path: Path to the images file
        labels_path: Path to the labels file
        num_images: Number of images to load

    Returns:
        List of (image, label) tuples
    """
    # Read labels
    with open(labels_path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        labels = np.fromfile(f, dtype=np.uint8)

    # Read images
    with open(images_path, "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.fromfile(f, dtype=np.uint8).reshape(len(labels), rows, cols)

    # Select a subset of images
    indices = list(range(len(labels)))
    random.seed(42)  # For reproducibility
    selected_indices = random.sample(indices, min(num_images, len(indices)))

    # Create a list of (image, label) tuples
    mnist_data = []
    for idx in selected_indices:
        img = Image.fromarray(images[idx])
        label = int(labels[idx])
        mnist_data.append((img, label))

    return mnist_data


# Load MNIST test images
mnist_test_images = load_mnist_images(
    "mnist/t10k-images.idx3-ubyte", "mnist/t10k-labels.idx1-ubyte", num_images=100
)

# Create global variables for gallery images and selected index
gallery_images = [img for img, _ in mnist_test_images]
gallery_labels = [f"Digit: {label}" for _, label in mnist_test_images]
selected_image_index = 0  # Default to first image

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


def image_to_tensor(image):
    """
    Convert a PIL image to a tensor for model input.

    Args:
        image: PIL Image

    Returns:
        Tensor representation of the image
    """
    if image is None:
        return None

    # Resize to 28x28 if needed
    if image.size != (28, 28):
        image = image.resize((28, 28), Image.Resampling.LANCZOS)

    # Convert to grayscale if it's not already
    if image.mode != "L":
        image = image.convert("L")

    img_array = np.array(image)

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

    # Format the metrics for display in the Gradio UI
    lmul_html = "<div style='text-align: left;'>"
    for key, value in lmul_metrics.items():
        lmul_html += f"<p><b>{key}:</b> {value}</p>"
    lmul_html += "</div>"

    ieee_html = "<div style='text-align: left;'>"
    for key, value in ieee_metrics.items():
        ieee_html += f"<p><b>{key}:</b> {value}</p>"
    ieee_html += "</div>"

    return (
        lmul_html,
        ieee_html,
    )


def update_selected_index(evt: gr.SelectData):
    """Update the selected image index when a gallery image is clicked"""
    global selected_image_index
    selected_image_index = evt.index
    return f"Selected Digit: {mnist_test_images[evt.index][1]}"


def predict_lmul(
    weight: str,
    activation: str,
    gr_progress=gr.Progress(track_tqdm=True),
):
    """Run the l-mul hardware simulation on the selected image"""
    global selected_image_index
    selected_image = gallery_images[selected_image_index]

    if selected_image is None:
        return labels_value

    if weight == "float8" and activation == "float8":
        activation = "bfloat16"
    config = CompiledAcceleratorConfig(
        array_size=8,
        activation_type=dtype_map[activation],
        weight_type=dtype_map[weight],
        multiplier=lmul_simple,
    )
    sim = CompiledAcceleratorSimulator(config, MODEL)

    x = image_to_tensor(selected_image).detach().numpy().flatten()
    probabilities = sim.predict(x)
    return {cls: float(prob) for cls, prob in zip(classes, probabilities)}


def predict_ieee(
    weight: str,
    activation: str,
    gr_progress=gr.Progress(track_tqdm=True),
):
    """Run the IEEE hardware simulation on the selected image"""
    global selected_image_index
    selected_image = gallery_images[selected_image_index]

    if selected_image is None:
        return labels_value

    if weight == "float8" and activation == "float8":
        activation = "bfloat16"
    config = CompiledAcceleratorConfig(
        array_size=8,
        activation_type=dtype_map[activation],
        weight_type=dtype_map[weight],
        multiplier=float_multiplier,
    )
    simulator = CompiledAcceleratorSimulator(config, MODEL)

    x = image_to_tensor(selected_image).detach().numpy().flatten()
    probabilities = simulator.predict(x)
    return {cls: float(prob) for cls, prob in zip(classes, probabilities)}


# ------------ Blocks UI Layout ------------ #


def create_app():
    with gr.Blocks(fill_height=False, fill_width=False, title=__file__) as demo:

        gr.Markdown("## MNIST Hardware Accelerator Simulation")

        with gr.Row(equal_height=False):
            with gr.Column(scale=3):
                # Create a gallery of MNIST images with square layout and preview enabled by default
                gallery = gr.Gallery(
                    value=[
                        (img, caption)
                        for img, caption in zip(gallery_images, gallery_labels)
                    ],
                    label="MNIST Test Images (Click to select an image)",
                    columns=[5, 5, 5, 5, 5, 5],  # Make it square across all breakpoints
                    rows=5,  # Match with columns for square layout
                    height=400,  # Fixed height to ensure square appearance
                    object_fit="contain",
                    allow_preview=True,
                    preview=True,  # Start in preview mode by default
                    elem_id="mnist_gallery",
                )

                # Display the selected digit
                selected_digit_text = gr.Markdown("Selected Digit: 0")

                # Add a button to run the hardware simulation
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

        # When an image is selected from the gallery, update the selected index
        gallery.select(
            fn=update_selected_index,
            outputs=selected_digit_text,
        )

        # Run hardware simulation when the button is clicked
        predict_btn.click(
            fn=predict_ieee,
            inputs=[weight_type_component, activation_type_component],
            outputs=ieee_predictions,
        )

        predict_btn.click(
            fn=predict_lmul,
            inputs=[weight_type_component, activation_type_component],
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
