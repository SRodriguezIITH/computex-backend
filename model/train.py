import os
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm  # Import tqdm for progress bars
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

# Import datasets and model
from model.architecture import CNN

os.system('clear')

# Set device (GPU if available)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print("\n\nCommencing Data Processing")

# Load the dataset
train_dataset_x = np.load("cache/train_dataset_x.npy")
train_dataset_y = np.load("cache/train_dataset_y.npy")
test_dataset_x = np.load("cache/test_dataset_x.npy")
test_dataset_y = np.load("cache/test_dataset_y.npy")

train_dataset_x = torch.tensor(train_dataset_x)
train_dataset_y = torch.tensor(train_dataset_y)
test_dataset_x = torch.tensor(test_dataset_x)
test_dataset_y = torch.tensor(test_dataset_y)

print(f"Training dataset size: {train_dataset_x.size()}")
print(f"Testing dataset size: {test_dataset_x.size()}")

# Create dataset and DataLoader
train_dataset = torch.utils.data.TensorDataset(train_dataset_x, train_dataset_y)
test_dataset = torch.utils.data.TensorDataset(test_dataset_x, test_dataset_y)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=True)

print("Data Processing Complete\n\n")

# Initialize model and move to device
print("Initializing model\n\n")
model = CNN().cuda()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Initialize GradScaler for mixed precision
scaler = torch.cuda.amp.GradScaler()

# Variables to track loss values and accuracy for plotting
train_losses = []
test_losses = []
accuracies = []

torch.cuda.empty_cache()

# Training function
def train(epoch):
    model.train()
    running_loss = 0.0
    os.system('clear')
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch} [Training]")
    
    for batch_idx, (data, targets) in progress_bar:
        
        data, targets = data.to(device), targets.to(device)
        data = data.to(torch.float32)
        targets = targets.to(torch.long)

        # Forward pass with mixed precision
        optimizer.zero_grad()
        
        with torch.amp.autocast("cuda"):  # Automatically casts operations to half precision
            output = model(data)
            loss = criterion(output, targets)
        
        running_loss += loss.item()

        # Backward pass with scaled gradients
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        progress_bar.set_postfix(loss=loss.item())
    
    # Calculate average loss for this epoch
    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    print(f"Epoch {epoch} - Average Training Loss: {avg_train_loss:.6f}")

    torch.save(model.state_dict(), "model.pth")

# Testing function
def test():
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    progress_bar = tqdm(test_loader, total=len(test_loader), desc="Testing")
    
    with torch.no_grad():
        for data, targets in progress_bar:
            os.system('clear')
            data, targets = data.to(device), targets.to(device)
            data = data.to(torch.float32)
            targets = targets.to(torch.long)
            
            output = model(data)
            test_loss += criterion(output, targets).item()
            
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(targets.view_as(pred)).sum().item()
            total += targets.size(0)
    
    test_loss /= len(test_loader)
    accuracy = 100. * correct / total
    print(f"Test Loss: {test_loss:.6f}, Accuracy: {correct}/{total} ({accuracy:.2f}%)")

    test_losses.append(test_loss)
    accuracies.append(accuracy)

# Train the model
num_epochs = 5
for epoch in range(1, num_epochs + 1):
    train(epoch)
    test()

# Plot Training and Validation Loss
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss', marker='o')
plt.plot(range(1, num_epochs + 1), test_losses, label='Validation Loss', marker='s')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss per Epoch')
plt.legend()
plt.grid(True)
plt.savefig('training_val_loss_plot.png')

# Plot Accuracy
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), accuracies, label='Accuracy (%)', color='green', marker='^')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.title('Validation Accuracy per Epoch')
plt.legend()
plt.grid(True)
plt.savefig('accuracy_plot.png')
