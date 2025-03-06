import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import math
import time


# Custom approximate matrix multiplication using lmul
def lmul_matmul(A: torch.Tensor, B: torch.Tensor, dtype=torch.float32):
    """
    Approximate matrix multiplication between A (m x n) and B (n x p)
    using bitwise operations to mimic multiplication.
    """
    if dtype == torch.float32:
        # reinterpret bits as uint32 then convert to int64 for arithmetic
        A_int = A.contiguous().view(torch.uint32).to(torch.int64)
        B_int = B.contiguous().view(torch.uint32).to(torch.int64)
        offset = 1064828928  # offset for float32
    elif dtype == torch.bfloat16:
        A_int = A.contiguous().view(torch.uint16).to(torch.int64)
        B_int = B.contiguous().view(torch.uint16).to(torch.int64)
        offset = 16248  # offset for bfloat16
    else:
        raise ValueError("Unsupported dtype")

    # A is (m, n) and B is (n, p).
    # Expand dims so that:
    # A_int becomes (m, n, 1) and B_int becomes (1, n, p)
    prod_int = A_int.unsqueeze(2) + B_int.unsqueeze(0) - offset  # shape: (m, n, p)

    # Convert the integer result back to floating point.
    if dtype == torch.float32:
        prod = prod_int.to(torch.uint32).view(torch.float32)
    else:
        prod = prod_int.to(torch.uint16).view(torch.bfloat16)

    # Sum over the inner dimension to complete the dot product.
    return prod.sum(dim=1)


# Custom linear layer that uses lmul-based matrix multiplication
class LmulLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, dtype=torch.float32):
        super(LmulLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dtype = dtype
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize weights similarly to nn.Linear.
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        # Compute the approximate matrix multiply:
        # Note: input shape is (batch, in_features)
        # weight.T shape is (in_features, out_features)
        out = lmul_matmul(input, self.weight.t(), self.dtype)
        if self.bias is not None:
            out = out + self.bias  # add bias as usual
        return out


# MLP model using our custom lmul-based linear layers
class LmulMLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, dtype=torch.float32):
        super(LmulMLP, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = LmulLinear(input_size, hidden_size, bias=True, dtype=dtype)
        self.relu = nn.ReLU()
        self.fc2 = LmulLinear(hidden_size, num_classes, bias=True, dtype=dtype)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# Setup: use float32 for this example.
dtype = torch.float32

# Instantiate the model.
# For MNIST: input size is 28x28 = 784, hidden layer of 128, output 10 classes.
model = LmulMLP(input_size=784, hidden_size=128, num_classes=10, dtype=dtype)
model.eval()  # set model to evaluation mode

model.load_state_dict(torch.load("models/mlp_mnist_fp32.pth", weights_only=True))

# Prepare the MNIST test dataset.
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ]
)
test_dataset = datasets.MNIST(
    root="./data", train=False, transform=transform, download=True
)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Run inference on the test dataset and measure accuracy.
correct = 0
total = 0
start_time = time.time()

with torch.no_grad():
    for images, labels in test_loader:
        # Ensure images are in the right dtype
        images = images.to(dtype)
        outputs = model(images)
        # Compute predictions
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted.cpu() == labels).sum().item()

end_time = time.time()
accuracy = correct / total * 100
inference_time = end_time - start_time

print(f"Test Accuracy: {accuracy:.2f}%")
print(f"Inference Time on Test Set: {inference_time:.2f} seconds")
