from tests import _PATH_DATA
import torch
import os
from src.data.make_dataset import Lego_Dataset
import pytest

@pytest.mark.parametrize('data_path,expected_data_shape', [
    (os.path.join(_PATH_DATA,'processed','train_dataset.pth'), torch.Size([3, 224, 224])),
    (os.path.join(_PATH_DATA,'processed','val_dataset.pth'), torch.Size([3, 224, 224])),
    (os.path.join(_PATH_DATA,'processed','test_dataset.pth'), torch.Size([3, 224, 224]))
])


def test_data_shape(data_path,expected_data_shape):
    dataset = torch.load(data_path)
    img,_ = dataset.__getitem__(0)
    assert img.shape == expected_data_shape


