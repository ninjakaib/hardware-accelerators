import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .util import get_pytorch_device  # progress bar for notebooks
from .mlp import model_factory

# from pytorch2tikz import Architecture


# Training function for one epoch
def train(model, device, train_loader, optimizer, criterion, epoch, num_epochs):
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}", leave=False)
    for batch_idx, (data, target) in enumerate(progress_bar):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch} Train Loss: {avg_loss:.4f}")


# Evaluation function
def evaluate(model, device, data_loader, criterion):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            loss = criterion(outputs, target)
            running_loss += loss.item()
            # Get predictions from the maximum value
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    avg_loss = running_loss / len(data_loader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


if __name__ == "main":
    device = get_pytorch_device()

    # Hyperparameters
    input_size = 28 * 28  # MNIST images are 28x28
    hidden_size = 128
    num_classes = 10
    num_epochs = 5
    batch_size = 32
    learning_rate = 0.001

    model = model_factory().to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # type: ignore
    criterion = nn.CrossEntropyLoss()

    # Print model's state_dict
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    # Print optimizer's state_dict
    print("Optimizer's state_dict:")
    for var_name in optimizer.state_dict():
        print(var_name, "\t", optimizer.state_dict()[var_name])

    # Data transformation: convert images to tensor and normalize them
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    # Load the MNIST training dataset (we will later perform CV on this)
    train_dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )

    # Train the model on the full training dataset
    print("Training on the full training set...")
    full_train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(1, num_epochs + 1):
        train(model, device, full_train_loader, optimizer, criterion, epoch, num_epochs)

    # Evaluate on the test set
    test_dataset = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    test_loss, test_accuracy = evaluate(model, device, test_loader, criterion)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%\n")

    model_dir = "models"
    pytorch_output = os.path.join(model_dir, "mlp_mnist.pth")
    onnx_output = os.path.join(model_dir, "mlp_mnist.onnx")

    model.eval()

    # Save pytorch model
    torch.save(model.state_dict(), pytorch_output)
    print(f"Model saved to 'mlp_mnist.pth'.\n")

    # Export the final model to ONNX format
    dummy_input = torch.randn(1, 1, 28, 28, device=device)
    torch.onnx.export(
        model,
        dummy_input,
        onnx_output,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=11,
    )
    print(f"Model exported to ONNX format at '{onnx_output}'.\n")
