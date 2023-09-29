import os
import sys
import torch
import pickle
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import timm
import subprocess
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

# Import LegoDataset class
# Get the parent directory (my_project) of the current script
current_script_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.abspath(os.path.join(current_script_directory, '..'))

# Add the parent directory to the Python module search path
sys.path.append(parent_directory)

# Now, you can perform a relative import from script1
from data.make_dataset import Lego_Dataset


# Load data
folder_path = 'data/processed/'  # Replace with the actual folder path
file_name = 'LEGO_torch_train_dataset.pickle'  # Replace with the actual file name
full_path = os.path.join(folder_path, file_name)

wd = os.getcwd()
path = os.path.join(wd,"C:/Users/dchro/Documents/MLOps/mlops_project/data/external/lego_dataset")

index = pd.read_csv( 'C:/Users/dchro/Documents/MLOps/mlops_project/data/external/lego_dataset/index.csv')
labels = index["class_id"]-1
files = index["path"]

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
trainset = Lego_Dataset(file_paths=files, path = path, labels=labels,transform=transform)


print(trainset[0][0].shape)
# train_loader = DataLoader(trainset, batch_size=32, shuffle=True)

print("Task ended")

# # Save the trained model
# torch.save(model.state_dict(), 'mobilenetv3_fine_tuned.pth')


