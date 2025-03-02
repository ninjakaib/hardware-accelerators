import torch
import numpy as np
from .mlp import MLP


def get_pytorch_device() -> torch.device:
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    return device


INPUT_DIM = 28 * 28  # MNIST images are 28x28
HIDDEN_SIZE = 128
OUTPUT_DIM = 10


def model_factory():
    return MLP(INPUT_DIM, HIDDEN_SIZE, OUTPUT_DIM)


def load_model(model_path: str, device: torch.device | None = None):
    if device is None:
        device = get_pytorch_device()
    model = model_factory()
    model.to(device)
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    return model


def softmax(x: np.ndarray):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)
