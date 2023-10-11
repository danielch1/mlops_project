# -*- coding: utf-8 -*-
import os
import string
from typing import Union

import hydra
import pandas as pd
import torch
from omegaconf import DictConfig
from torchvision import transforms

from src import _PROJECT_ROOT
from src.data.Lego_Dataset import Lego_Dataset

# processes the data from the external data and creates a Dataset of type Lego Dataset,
# adds various transformations for data augmentation
# samples indices from the train data_set to create a split between training and validation set
# and saves both files under processed


def make_dataset(config: DictConfig, dataset_type: string) -> Lego_Dataset:
    index_file = get_index_file(dataset_type)
    data_labels = index_file["class_id"] - 1
    data_files = index_file["path"]
    dataset = Lego_Dataset(
        file_paths=data_files,
        labels=data_labels,
        transform=get_transform(config, dataset_type=dataset_type),
    )
    # save_dataset(dataset= dataset, dataset_type = dataset_type)
    return dataset


# implements the split between training and validation data
# ToDo add Typing return value
def train_val_split(
    input_path_index: pd.DataFrame,
) -> Union[pd.DataFrame, pd.DataFrame]:
    # takes the path and label information from the provided index.csv file
    index = pd.read_csv(input_path_index)
    # Train Validation Split taking 3/4 of the training set as validation
    train_index = index.sample(int(0.8 * len(index.index)), random_state=42)

    remaining_indices = list(set(index.index) - set(train_index.index))
    # Create a new DataFrame with the remaining indices
    train_index.reset_index(inplace=True, drop=True)
    val_index = index.loc[remaining_indices].reset_index(drop=True)

    return train_index, val_index


# converts the label to the actual name of the minifigure
def convert_label(label: int) -> string:
    meta_data = pd.read_csv(
        os.path.join(_PROJECT_ROOT, "data", "external", "lego_dataset", "metadata.csv")
    )

    return (
        meta_data["minifigure_name"].loc[meta_data["class_id"] == label + 1].to_string()
    )


def get_transform(config: DictConfig, dataset_type: string):
    augmentation = []

    dataset_type = dataset_type + "_transforms"
    # Check if the dataset type is present in the configuration
    if dataset_type in config.augmentation.keys():
        # Access the transformations for the current dataset type
        dataset_transforms = config.augmentation[dataset_type]

        # Iterate over the transformations defined in the configuration
        for _, conf in dataset_transforms.items():
            if "_target_" in conf:
                # Instantiate and append the augmentation transform
                augmentation.append(hydra.utils.instantiate(conf))
    else:
        raise ValueError("Invalid dataset_type. It must be 'train', 'val', or 'test'.")

    # Create a composition of all augmentation transforms for the current dataset type
    return transforms.Compose(augmentation)


def get_index_file(dataset_type) -> pd.DataFrame:
    if dataset_type == "train":
        index, _ = train_val_split(
            os.path.join(_PROJECT_ROOT, "data", "external", "lego_dataset", "index.csv")
        )
    elif dataset_type == "val":
        _, index = train_val_split(
            os.path.join(_PROJECT_ROOT, "data", "external", "lego_dataset", "index.csv")
        )
    elif dataset_type == "test":
        index = pd.read_csv(
            os.path.join(_PROJECT_ROOT, "data", "external", "lego_dataset", "test.csv")
        )
    else:
        raise ValueError("Invalid dataset_type. It must be 'train', 'val', or 'test'.")
    return index


# saves processed data as tensors in a file
def save_dataset(dataset: Lego_Dataset, dataset_type: string) -> None:
    # Create an empty dictionary to store your data points.
    data_dict = {"images": [], "labels": []}

    # Loop through the dataset and collect image tensors and labels.
    for idx in range(len(dataset)):
        image_tensor, label = dataset[idx]  # Get image tensor and label.

        # Append the image tensor and label to the dictionary.
        data_dict["images"].append(image_tensor)
        data_dict["labels"].append(label)

        # Specify the directory and filename where you want to save the dictionary.
        save_path = os.path.join(
            _PROJECT_ROOT, "data", "processed", dataset_type + "_dataset.pth"
        )

        # Save the dictionary containing all data points to a single file.
        torch.save(data_dict, save_path)
