import torch.nn as nn

INPUT_DIM = 28 * 28  # MNIST images are 28x28
HIDDEN_SIZE = 128
OUTPUT_DIM = 10


# Define the MLP model with one hidden layer using ReLU
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        out = self.fc2(x)
        return out


def model_factory():
    return MLP(INPUT_DIM, HIDDEN_SIZE, OUTPUT_DIM)
