import torch
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


def load_model(model_path: str):
    device = get_pytorch_device()
    model = model_factory()
    model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model
