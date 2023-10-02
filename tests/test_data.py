from tests import _PATH_DATA
import torch
import os
from src.data.make_dataset import Lego_Dataset
import pytest
import numpy as np

@pytest.mark.parametrize('data_path,expected_data_shape', [
    (os.path.join(_PATH_DATA,'processed','train_dataset.pth'), torch.Size([3, 224, 224])),
    (os.path.join(_PATH_DATA,'processed','val_dataset.pth'), torch.Size([3, 224, 224])),
    (os.path.join(_PATH_DATA,'processed','test_dataset.pth'), torch.Size([3, 224, 224]))
])


def test_data_shape(data_path,expected_data_shape):
    #load dataset from file
    dataset = torch.load(data_path)
    dataset.set_path()
    #iterate through every image in dataset and compare to the expected shape
    check_shape = np.array([dataset.__getitem__(i)[0].shape == expected_data_shape for i in range(len(dataset))]).sum()
    #show that every image has the correct shape
    assert check_shape == len(dataset)
