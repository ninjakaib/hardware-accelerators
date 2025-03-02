import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .util import get_pytorch_device


# Define the MLP model (unchanged)
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


# Helper function: adjust data to match the target dtype
def convert_input(data, precision):
    if precision == "fp16":
        return data.half()
    elif precision == "bf16":
        return data.to(torch.bfloat16)
    elif precision == "fp8":
        # Note: torch.float8_e4m3 is experimental and may not be available
        return data.to(torch.float8_e4m3fn)
    return data  # fp32 (no conversion)


# Training for one epoch
def train_epoch(model, device, train_loader, optimizer, criterion, precision):
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(train_loader, desc="Training", leave=False)
    for data, target in progress_bar:
        # Convert inputs to the desired precision (targets remain integer)
        data = convert_input(data, precision)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(data)
        loss = criterion(outputs, target)
        
        # Check for NaN and skip problematic batches
        if torch.isnan(loss):
            print("NaN loss detected in batch, skipping...")
            continue
            
        # Backward and optimize with gradient clipping
        loss.backward()
        
        # Apply gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        running_loss += loss.item()
    
    if len(train_loader) > 0:
        return running_loss / len(train_loader)
    return 0.0


# Evaluation loop
def evaluate(model, device, test_loader, criterion, precision):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = convert_input(data, precision)
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            loss = criterion(outputs, target)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    avg_loss = total_loss / len(test_loader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


# Main training function for a given precision variant
def train_model(
    precision,
    batch_size=32,
    hidden_size=128,
    num_epochs=5,
    learning_rate=0.001,
    optimizer_name="adam",
    weight_decay=0,
    eps=1e-4,
    model_save_path=None
):
    print(f"\nTraining in {precision.upper()} mode:")
    device = get_pytorch_device()

    # Data transformation: images are loaded as FP32 by default
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    train_dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Hyperparameters
    input_size = 28 * 28  # MNIST images are 28x28
    num_classes = 10
    
    # Create the model and send to device
    model = MLP(input_size, hidden_size, num_classes).to(device)

    # Convert the model to the target precision (natively)
    if precision == "fp16":
        model = model.to(torch.float16)
        # Use a smaller learning rate for half precision if not explicitly specified
        if learning_rate == 0.001:  # If using the default value
            learning_rate = 1e-4  # Lower learning rate for stability
    elif precision == "bf16":
        model = model.to(torch.bfloat16)
    elif precision == "fp8":
        # Ensure your PyTorch build/hardware supports float8_e4m3; otherwise, this will error.
        model = model.to(torch.float8_e4m3fn)
    # else, fp32 is already the default
    
    # Select optimizer based on user input
    if optimizer_name.lower() == "adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, eps=eps, weight_decay=weight_decay)
    elif optimizer_name.lower() == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    elif optimizer_name.lower() == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, eps=eps, weight_decay=weight_decay)
    else:
        print(f"Unknown optimizer: {optimizer_name}, defaulting to Adam")
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, eps=eps, weight_decay=weight_decay)
    
    criterion = nn.CrossEntropyLoss()
    
    print(f"Training with: batch_size={batch_size}, hidden_size={hidden_size}, " 
          f"epochs={num_epochs}, lr={learning_rate}, optimizer={optimizer_name}")

    # Training loop
    for epoch in range(1, num_epochs + 1):
        train_loss = train_epoch(
            model, device, train_loader, optimizer, criterion, precision
        )
        
        # Check for NaN loss
        if torch.isnan(torch.tensor([train_loss])):
            print(f"NaN detected in epoch {epoch}, reducing learning rate")
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.5
        
        print(f"Epoch {epoch} Train Loss: {train_loss:.4f}")

    # Evaluation on test set
    test_loss, test_accuracy = evaluate(
        model, device, test_loader, criterion, precision
    )
    print(
        f"{precision.upper()} Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%"
    )

    # Optionally, save the model
    if model_save_path:
        save_path = model_save_path
    else:
        model_dir = "models"
        os.makedirs(model_dir, exist_ok=True)
        save_path = os.path.join(model_dir, f"mlp_mnist_{precision}.pth")
    
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}\n")


# Main script to train a model in a specific precision
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train MNIST model in a specific precision")
    parser.add_argument(
        "--dtype", 
        type=str,
        default="fp32",
        choices=["fp32", "fp16", "bf16", "fp8"],
        help="Precision type to train in (fp32, fp16, bf16, fp8)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training"
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=128,
        help="Hidden layer size for MLP"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate"
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        choices=["adam", "sgd", "adamw"],
        help="Optimizer to use for training"
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0,
        help="Weight decay (L2 penalty)"
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-4,
        help="Epsilon for Adam optimizer"
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="Path to save the trained model"
    )
    
    args = parser.parse_args()
    
    try:
        train_model(
            precision=args.dtype,
            batch_size=args.batch_size,
            hidden_size=args.hidden_size,
            num_epochs=args.epochs,
            learning_rate=args.lr,
            optimizer_name=args.optimizer,
            weight_decay=args.weight_decay,
            eps=args.eps,
            model_save_path=args.save_path
        )
    except Exception as e:
        print(f"Error training {args.dtype.upper()} model: {e}")
