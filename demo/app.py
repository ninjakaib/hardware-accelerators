import gradio as gr
from gradio.components.image_editor import EditorValue
import numpy as np
import torch
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from hardware_accelerators.nn import model_factory

# Load the trained model
model_path = "mlp_mnist.pth"
model = model_factory()
model.load_state_dict(torch.load(model_path, map_location=torch.device("mps")))
model.eval()


# Define the prediction function for the gradio app
def predict_image(image: EditorValue):
    image = image["composite"]
    # Preprocessing: convert image to grayscale, resize, tensor, and normalize
    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            # transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    # Process the image
    tensor_image = transform(image).unsqueeze(0)  # Add batch dimension
    print(tensor_image)
    # Forward pass through the model
    with torch.no_grad():
        logits = model(tensor_image)
        probabilities = torch.softmax(logits, dim=1).squeeze(0)

    # Map output to a dictionary with keys "zero" to "nine"
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
    result = {cls: float(prob) for cls, prob in zip(classes, probabilities)}
    return result


def get_image_data(sketchpad: EditorValue):
    # Convert to grayscale and resize to 28x28
    bg = sketchpad["background"]
    layers = sketchpad["layers"]
    comp = sketchpad["composite"]
    # print(f"bg: {np.max(bg.flatten())}")s
    # print(f"layers: {np.max(np.array(layers).flatten())}")
    # print(f"comp: {np.max(comp.flatten())}")

    image = sketchpad["composite"]
    # image = image.convert("L")
    # image = image.resize((28, 28))
    image = image.resize((28, 28), Image.Resampling.LANCZOS)

    # return image  # .resize((600, 600))

    # Convert to numpy array and normalize
    img_array = np.array(image)
    # img_array = img_array / 255.0

    img_array = np.transpose(img_array, (2, 0, 1))[-1]

    # Preprocessing: convert image to grayscale, resize, tensor, and normalize
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    # Process the image
    tensor_image = transform(img_array).unsqueeze(0)  # Add batch dimension
    print(tensor_image)
    # Forward pass through the model
    with torch.no_grad():
        logits = model(tensor_image)
        probabilities = torch.softmax(logits, dim=1).squeeze(0)

    # Map output to a dictionary with keys "zero" to "nine"
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
    result = {cls: float(prob) for cls, prob in zip(classes, probabilities)}
    return result
    # Convert to tensor format expected by model
    # Add batch and channel dimensions
    # img_tensor = torch.FloatTensor(img_array).unsqueeze(0).unsqueeze(0)

    # return img_tensor


with gr.Blocks(fill_height=True) as demo:
    with gr.Row():
        with gr.Column():
            sketchpad = gr.Sketchpad(
                height=800,
                width=800,
                # value=EditorValue(
                #     background=Image.new("L", (800, 800), (255)),
                #     layers=None,
                #     composite=None,
                # ),
                # image_mode="L",
                # brush=gr.Brush(colors=[0], color_mode="fixed"),
                type="pil",  # Changed to PIL
                transforms=(),
                # layers=False,
                fixed_canvas=True,
                container=False,
            )
        with gr.Column():
            output = gr.Label(
                label="Class Probabilities",
                value={
                    "zero": 0.0,
                    "one": 0.0,
                    "two": 0.0,
                    "three": 0.0,
                    "four": 0.0,
                    "five": 0.0,
                    "six": 0.0,
                    "seven": 0.0,
                    "eight": 0.0,
                    "nine": 0.0,
                },
            )
            # data = gr.DataFrame(
            #     type="numpy",
            # )
            data = gr.Image(type="pil", image_mode="RGB")

    sketchpad.input(
        fn=get_image_data,
        inputs=sketchpad,
        outputs=output,
    )
    # .then(predict_image, inputs=sketchpad, outputs=output)

demo.queue()
demo.launch(share=False)
