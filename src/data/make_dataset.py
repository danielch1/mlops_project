# -*- coding: utf-8 -*-
import pandas as pd
import torch
import os
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import logging
from torchvision import transforms
import sys
import torch
from PIL import Image

class Lego_Dataset(torch.utils.data.Dataset):
    def __init__(self, file_paths, path, labels, transform=None):
        """
        Args:
            file_paths (list): List of file paths for the images.
            labels (list): List of corresponding labels.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform
        self.path = path

    def __len__(self):
        return len(self.file_paths)


    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        image = Image.open(os.path.join(self.path,img_path)).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]

        return image, label




def get_data():

    current_script_directory = os.path.dirname(os.path.abspath(__file__))
    parent_directory = os.path.abspath(os.path.join(current_script_directory, '..'))
    root_directory = os.path.abspath(os.path.join(parent_directory, '..'))
    sys.path.append(parent_directory)


    
    data_path = os.path.join(root_directory,"data", "external", "lego_dataset")

    index = pd.read_csv(os.path.join(data_path, 'index.csv'))


    #Train Validation Split
    train_index = index.sample(int(0.75*len(index.index)))

    remaining_indices = list(set(index.index) - set(train_index.index))
    # Create a new DataFrame with the remaining indices
    train_index.reset_index(inplace= True, drop=True)
    val_index = index.loc[remaining_indices].reset_index(drop=True)

    train_labels = train_index["class_id"]-1
    train_files = train_index["path"]

    val_labels = val_index["class_id"]-1
    val_files = val_index["path"]


    train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),  # Random crop and resize to 224x224
    transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
    transforms.RandomRotation(degrees=30),  # Randomly rotate the image up to 30 degrees
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Randomly translate the image
    transforms.RandomPerspective(distortion_scale=0.5, p=0.5),  # Apply perspective transformation
    transforms.ToTensor(),  # Convert the image to a PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet statistics
    ])

    # Define transforms for validation and test data (typically no augmentation)
    val_transforms = transforms.Compose([
        transforms.Resize(256),  # Resize to 256x256
        transforms.CenterCrop(224),  # Center crop to 224x224
        transforms.ToTensor(),  # Convert the image to a PyTorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet statistics
    ])


    return Lego_Dataset(file_paths=train_files, path = data_path, labels=train_labels,transform=train_transforms),Lego_Dataset(file_paths=val_files, path = data_path, labels=val_labels,transform=val_transforms)


def get_test_Data():
    current_script_directory = os.path.dirname(os.path.abspath(__file__))
    parent_directory = os.path.abspath(os.path.join(current_script_directory, '..'))
    root_directory = os.path.abspath(os.path.join(parent_directory, '..'))
    sys.path.append(parent_directory)


    
    data_path = os.path.join(root_directory,"data", "external", "lego_dataset")

    index = pd.read_csv(os.path.join(data_path, 'test.csv'))
    
