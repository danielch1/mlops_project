import os
import torch
import torch.nn as nn
import torch.optim as optim
import timm
import subprocess
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

# Load data

folder_path = 'data/processed/'  # Replace with the actual folder path
file_name = 'LEGO_torch_train_dataset.pt'  # Replace with the actual file name
full_path = os.path.join(folder_path, file_name)

# Check if the file dataset exists in the location otherwise run the script to create it
if not os.path.exists(full_path):
    # Run the script to create the dataset
    subprocess.call(['python', 'src/data/make_dataset.py'])
    print("Dataset created")

# Load the dataset
trainset = torch.load(full_path)
train_loader = DataLoader(trainset, batch_size=32, shuffle=True)

print("Task ended")

# # Save the trained model
# torch.save(model.state_dict(), 'mobilenetv3_fine_tuned.pth')


