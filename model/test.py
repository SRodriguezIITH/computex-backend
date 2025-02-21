import os
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm  # Import tqdm for progress bars
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from model.architecture import CNN

# Initialize device, either CUDA or CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load the trained model
model = CNN().cuda()
model.load_state_dict(torch.load("model.pth"))
model.eval()

# Load the test dataset
test_dataset_x = np.load("cache/test_dataset_x.npy")
test_dataset_y = np.load("cache/test_dataset_y.npy")

test_dataset_x = torch.tensor(test_dataset_x)
test_dataset_y = torch.tensor(test_dataset_y)

print(f"Testing dataset size: {test_dataset_x.size()}")

# Create dataset and DataLoader
test_dataset = torch.utils.data.TensorDataset(test_dataset_x, test_dataset_y)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=True)

# Loss function
criterion = nn.CrossEntropyLoss()

# Initialize lists for tracking the loss and accuracy
test_losses = []
accuracies = []

# Define the test function
def test():
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    progress_bar = tqdm(test_loader, total=len(test_loader), desc="Testing")
    
    with torch.no_grad():
        for data, targets in progress_bar:
            os.system('clear')  # Clear the terminal screen for each batch
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

            test_losses.append(test_loss)
            accuracies.append(accuracy)


    print(f"Test Loss: {test_loss:.6f}, Accuracy: {correct}/{total} ({accuracy:.2f}%)")

    
# Run the test function
test()

# Plotting the test loss and accuracy
fig, ax1 = plt.subplots()

# Plot loss
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss', color='tab:blue')
ax1.plot(test_losses, color='tab:blue', label='Test Loss')
ax1.tick_params(axis='y', labelcolor='tab:blue')

# Create a second y-axis for accuracy
ax2 = ax1.twinx()
ax2.set_ylabel('Accuracy (%)', color='tab:red')
ax2.plot(accuracies, color='tab:red', label='Accuracy')
ax2.tick_params(axis='y', labelcolor='tab:red')

fig.tight_layout()
plt.legend(loc="upper left")
plt.show()
