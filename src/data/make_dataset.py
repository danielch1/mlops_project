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
import random
from src.data.Lego_Dataset import Lego_Dataset
from src import _PROJECT_ROOT


# processes the data from the external data and creates a Dataset of type Lego Dataset, adds various transformations for data augmentation
# samples indices from the train data_set to create a split between training and validation set and saves both files under processed
def make_train_data():
    # the index dataframe provides the path as well as the naming of the classes
    index = pd.read_csv(
        os.path.join(_PROJECT_ROOT, "data", "external", "lego_dataset", "index.csv")
    )

    # Train Validation Split taking 3/4 of the training set as validation
    train_index = index.sample(int(0.75 * len(index.index)), random_state=42)

    remaining_indices = list(set(index.index) - set(train_index.index))
    # Create a new DataFrame with the remaining indices
    train_index.reset_index(inplace=True, drop=True)
    val_index = index.loc[remaining_indices].reset_index(drop=True)

    # since the first label is 1 and torch starts from 0 the labels need to be corrected for that
    train_labels = train_index["class_id"] - 1
    train_files = train_index["path"]

    val_labels = val_index["class_id"] - 1
    val_files = val_index["path"]

    # Define transforms for validation and test data (typically no augmentation)
    train_transforms = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),  # Random crop and resize to 224x224
            transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
            transforms.RandomRotation(
                degrees=30
            ),  # Randomly rotate the image up to 30 degrees
            transforms.RandomAffine(
                degrees=0, translate=(0.1, 0.1)
            ),  # Randomly translate the image
            transforms.RandomPerspective(
                distortion_scale=0.5, p=0.5
            ),  # Apply perspective transformation
            transforms.ToTensor(),  # Convert the image to a PyTorch tensor
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),  # ImageNet statistics
        ]
    )

    val_transforms = transforms.Compose(
        [
            transforms.Resize(256),  # Resize to 256x256
            transforms.CenterCrop(224),  # Center crop to 224x224
            transforms.ToTensor(),  # Convert the image to a PyTorch tensor
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),  # ImageNet statistics
        ]
    )

    # saving both Datasets to the processed directory within the project
    torch.save(
        Lego_Dataset(
            file_paths=train_files,
            labels=train_labels,
            transform=train_transforms,
        ),
        os.path.join(_PROJECT_ROOT, "data", "processed", "train_dataset.pth"),
    )
    torch.save(
        Lego_Dataset(
            file_paths=val_files,
            labels=val_labels,
            transform=val_transforms,
        ),
        os.path.join(_PROJECT_ROOT, "data", "processed", "val_dataset.pth"),
    )

    # Creates the test data set from the external files and saves the torch Dataset to processed


def make_test_Data():
    test_index = pd.read_csv(
        os.path.join(_PROJECT_ROOT, "data", "external", "lego_dataset", "test.csv")
    )

    test_labels = test_index["class_id"] - 1
    test_files = test_index["path"]

    test_transforms = transforms.Compose(
        [
            transforms.Resize(256),  # Resize to 256x256
            transforms.CenterCrop(224),  # Center crop to 224x224
            transforms.ToTensor(),  # Convert the image to a PyTorch tensor
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),  # ImageNet statistics
        ]
    )

    torch.save(
        Lego_Dataset(
            file_paths=test_files,
            labels=test_labels,
            transform=test_transforms,
        ),
        os.path.join(_PROJECT_ROOT, "data", "processed", "test_dataset.pth"),
    )


# converts the label to the actual name of the minifigure
def convert_label(label):
    meta_data = pd.read_csv(
        os.path.join(_PROJECT_ROOT, "data", "external", "lego_dataset", "metadata.csv")
    )
    return meta_data["minifigure_name"].loc[meta_data["class_id"] == label].to_string()


# where to execute this from?
if __name__ == "__main__":
    make_train_data()
    make_test_Data()
