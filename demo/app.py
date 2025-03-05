import sys
import gradio as gr
from gradio.components.image_editor import EditorValue
import numpy as np
import torch
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import tqdm
import pyrtl

sys.path.append(".")
from hardware_accelerators.nn.util import load_model, softmax
from hardware_accelerators.simulation.matrix_utils import (
    bias_trick,
    count_total_gemv_tiles,
    generate_gemv_tiles,
)
from hardware_accelerators.rtllib.adders import float_adder
from hardware_accelerators.rtllib.multipliers import float_multiplier
from hardware_accelerators.dtypes.bfloat16 import BF16
from hardware_accelerators.dtypes.float8 import Float8
from hardware_accelerators.nn import model_factory, get_pytorch_device
from hardware_accelerators.rtllib import (
    CompiledAcceleratorConfig,
    AcceleratorConfig,
    Accelerator,
    lmul_fast,
    float_multiplier,
)
from hardware_accelerators.simulation import CompiledAcceleratorSimulator


# ------------ CONSTANTS ------------ #

# Load the trained model
model = load_model("models/mlp_mnist.pth")
model.eval()

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

accelerator_dtypes = ["float8", "bfloat16"]
# accelerator_dtypes = ["float8", "float16", "bfloat16", "float32"]

dtype_map = {"float8": Float8, "bfloat16": BF16}

default_config = {
    "activations_dtype": "bfloat16",
    "weights_dtype": "bfloat16",
    "size": 4,
    "multiplication": "IEEE 754",
}

mult_map = {
    "IEEE 754": float_multiplier,
    "l-mul": lmul_fast,
}


# ------------ Event Listener Functions ------------ #


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
    tensor_image = transform(img_array).unsqueeze(0)  # Add batch dimension
    return tensor_image


def torch_predict(sketchpad: EditorValue):
    tensor_image = image_to_tensor(sketchpad)
    with torch.no_grad():
        logits = model(tensor_image)
        probabilities = torch.softmax(logits, dim=1).squeeze(0)
    result = {cls: float(prob) for cls, prob in zip(classes, probabilities)}
    return result


def update_accelerator_config(
    activations_dtype: str,
    weights_dtype: str,
    size: int,
    multiplication: str,
) -> CompiledAcceleratorConfig:
    config = CompiledAcceleratorConfig(
        array_size=size,
        activation_type=BF16,
        weight_type=BF16,
        multiplier=float_multiplier,
    )

    # Triggered by run simulation button
    print("update_accelerator_config fn called")
    print(config)

    return config


def sim_predict_progress(
    sketchpad: EditorValue,
    config: CompiledAcceleratorConfig,
    gr_progress=gr.Progress(track_tqdm=True),
):
    pyrtl.reset_working_block()
    simulator = CompiledAcceleratorSimulator(config)
    chunk_size = config.array_size

    x = image_to_tensor(sketchpad).detach().numpy().flatten()
    probabilities = simulator.predict(x)
    return {cls: float(prob) for cls, prob in zip(classes, probabilities)}


# ------------ Blocks UI Layout ------------ #

with gr.Blocks(fill_height=False) as demo:

    accelerator_config = gr.State()

    gr.Markdown("## Draw a digit to see the model's prediction")
    with gr.Row(equal_height=True):
        with gr.Column():
            sketchpad = gr.Sketchpad(
                # label="Draw a digit",
                type="pil",  # Changed to PIL
                transforms=(),
                layers=False,
                canvas_size=(400, 400),
            )

            with gr.Row():
                predict_btn = gr.Button("Run Hardware Simulation", variant="primary")

            # with gr.Accordion("Accelerator Configuration", open=True):
            with gr.Group():
                weight_dtype_component = gr.Radio(
                    label="Weights d-type",
                    choices=accelerator_dtypes,
                    value=default_config["weights_dtype"],
                    interactive=True,
                )
                activation_dtype_component = gr.Radio(
                    label="Activations d-type",
                    choices=accelerator_dtypes,
                    value=default_config["activations_dtype"],
                    interactive=True,
                )
                systolic_array_size_component = gr.Slider(
                    label="Systolic Array Size",
                    info="Large values will significantly slow down the simulation",
                    minimum=2,
                    maximum=16,
                    step=1,
                    value=default_config["size"],
                    interactive=True,
                )
                multiply_component = gr.Radio(
                    label="Multiplication Type",
                    choices=["IEEE 754", "l-mul"],
                    value=default_config["multiplication"],
                    interactive=True,
                )

        with gr.Column():
            pytorch_output = gr.Label(
                label="Pytorch Ground Truth Predictions", value=labels_value
            )

            sim_output = gr.Label(
                label="Hardware Simulator Predictions", value=labels_value
            )

    # ------------ Event Listeners ------------ #

    sketchpad.input(
        fn=torch_predict,
        inputs=sketchpad,
        outputs=pytorch_output,
    )

    # TODO: implement simulator_predict
    predict_btn.click(
        fn=update_accelerator_config,
        inputs=[
            activation_dtype_component,
            weight_dtype_component,
            systolic_array_size_component,
            multiply_component,
        ],
        outputs=accelerator_config,
    ).then(
        fn=sim_predict_progress,
        inputs=[sketchpad, accelerator_config],
        outputs=sim_output,
    )

    # gr.on(
    #     fn=update_accelerator_config,
    #     inputs=[
    #         activation_dtype_component,
    #         weight_dtype_component,
    #         systolic_array_size_component,
    #         multiply_component,
    #     ],
    #     outputs=accelerator_config,
    # )

    # ------------

if __name__ == "__main__":
    demo.queue()
    demo.launch(share=False)
