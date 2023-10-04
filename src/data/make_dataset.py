# -*- coding: utf-8 -*-
import pandas as pd
import torch
import os
from torchvision import transforms
import torch
import random
from src.data.Lego_Dataset import Lego_Dataset
from src import _PROJECT_ROOT
import string
from typing import Union
from omegaconf import DictConfig
import hydra


# processes the data from the external data and creates a Dataset of type Lego Dataset, adds various transformations for data augmentation
# samples indices from the train data_set to create a split between training and validation set and saves both files under processed
def make_dataset(file_path_index : pd.DataFrame,transform : transforms) -> Lego_Dataset:
    data_labels = file_path_index["class_id"] - 1
    data_files = file_path_index["path"]
    
    return Lego_Dataset(file_paths=data_files,labels=data_labels,transform=transform)
    

#implements the split between training and validation data
#ToDo add Typing return value
def train_val_split(input_path_index : pd.DataFrame) -> Union[pd.DataFrame, pd.DataFrame]:
    # takes the path and label information from the provided index.csv file
    index = pd.read_csv(input_path_index)
    # Train Validation Split taking 3/4 of the training set as validation
    train_index = index.sample(int(0.75 * len(index.index)), random_state=42)

    remaining_indices = list(set(index.index) - set(train_index.index))
    # Create a new DataFrame with the remaining indices
    train_index.reset_index(inplace=True, drop=True)
    val_index = index.loc[remaining_indices].reset_index(drop=True)

    return train_index,val_index

# converts the label to the actual name of the minifigure
def convert_label(label : int) -> string:
    meta_data = pd.read_csv(
        os.path.join(_PROJECT_ROOT, "data", "external", "lego_dataset", "metadata.csv")
    )
    return meta_data["minifigure_name"].loc[meta_data["class_id"] == label].to_string()


# where to execute this from?
@hydra.main(config_path='../configs/', config_name='main.yaml')
def create_data(config: DictConfig = ):

    augmentation : List[torch.nn.Module] = []
    if "augmentation" in config:
        for _, conf in config.augmentation.items():
            if "_target_" in conf:
                preprocess.append(hydra.utils.instantiate(conf))

    augmentation_compose = transforms.Compose(augmentation)

    #defining the correct input and output paths for traing, validation and test set preprocessing
    train_input_index,val_input_index = train_val_split(os.path.join(_PROJECT_ROOT, "data", "external", "lego_dataset", "index.csv"))
    train_output_path = os.path.join(_PROJECT_ROOT, "data", "processed", "train_dataset.pth")

    val_output_path = os.path.join(_PROJECT_ROOT, "data", "processed", "val_dataset.pth")

    test_input_index = pd.read_csv(os.path.join(_PROJECT_ROOT, "data", "external", "lego_dataset", "test.csv"))
    test_output_path = os.path.join(_PROJECT_ROOT, "data", "processed", "test_dataset.pth")

    #should be managed via config management
    train_transforms = transforms.Compose()
    #     [
    #         transforms.RandomResizedCrop(224),  # Random crop and resize to 224x224
    #         transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
    #         transforms.RandomRotation(
    #             degrees=30
    #         ),  # Randomly rotate the image up to 30 degrees
    #         transforms.RandomAffine(
    #             degrees=0, translate=(0.1, 0.1)
    #         ),  # Randomly translate the image
    #         transforms.RandomPerspective(
    #             distortion_scale=0.5, p=0.5
    #         ),  # Apply perspective transformation
    #         transforms.ToTensor(),  # Convert the image to a PyTorch tensor
    #         transforms.Normalize(
    #             mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    #         ),
    #     ]
    # )
    # val_test_transforms = transforms.Compose(
    #     [
    #         transforms.Resize(256),  # Resize to 256x256
    #         transforms.CenterCrop(224),  # Center crop to 224x224
    #         transforms.ToTensor(),  # Convert the image to a PyTorch tensor
    #         transforms.Normalize(
    #             mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    #         ),  # ImageNet statistics
    #     ]
    # )

    torch.save(make_dataset(train_input_index,train_transforms),train_output_path)
    torch.save(make_dataset(val_input_index,val_test_transforms),val_output_path)
    torch.save(make_dataset(test_input_index,val_test_transforms),test_output_path)

