from tests import _PATH_DATA
import torch
import os
from src.data.make_dataset import Lego_Dataset
from src.data.make_dataset import make_dataset, train_val_split
import pytest
import numpy as np
import pandas as pd
from torchvision import transforms


@pytest.mark.parametrize(
    "data_path,expected_data_shape",
    [
        (
            os.path.join(_PATH_DATA, "processed", "train_dataset.pth"),
            torch.Size([3, 224, 224]),
        ),
        (
            os.path.join(_PATH_DATA, "processed", "val_dataset.pth"),
            torch.Size([3, 224, 224]),
        ),
        (
            os.path.join(_PATH_DATA, "processed", "test_dataset.pth"),
            torch.Size([3, 224, 224]),
        ),
    ],
)
def test_data_shape(data_path, expected_data_shape):
    # load dataset from file
    dataset = torch.load(data_path)
    dataset.set_path()
    # iterate through every image in dataset and compare to the expected shape
    check_shape = np.array(
        [
            dataset.__getitem__(i)[0].shape == expected_data_shape
            for i in range(len(dataset))
        ]
    ).sum()
    # show that every image has the correct shape
    assert check_shape == len(dataset)


def test_make_dataset():
    val_test_transforms = transforms.Compose(
        [
            transforms.Resize(256),  # Resize to 256x256
            transforms.CenterCrop(224),  # Center crop to 224x224
            transforms.ToTensor(),  # Convert the image to a PyTorch tensor
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),  # ImageNet statistics
        ]
    )

    train_index,_ = train_val_split(os.path.join(_PATH_DATA, "external", "lego_dataset", "index.csv"))
    assert len(make_dataset(train_index.sample(1, random_state=42), val_test_transforms)) == 1