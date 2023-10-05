from tests import _PATH_DATA
import torch
from src.data.make_dataset import Lego_Dataset
from src.data.make_dataset import make_dataset
import pytest
import numpy as np
from hydra import initialize, compose


@pytest.mark.parametrize(
    "dataset_type,expected_data_shape",
    [
        (
            "train",
            torch.Size([3, 224, 224]),
        ),
        (
            "val",
            torch.Size([3, 224, 224]),
        ),
        (
            "test",
            torch.Size([3, 224, 224]),
        ),
    ],
)

#@hydra.main(config_path="../../config/", config_name="main.yaml")
def data_shape_test(dataset_type, expected_data_shape):
    with initialize(version_base=None, config_path='../config/'):
        cfg = compose(config_name= 'main.yaml')
        # load dataset from file
        dataset = make_dataset(cfg,dataset_type)
        # iterate through every image in dataset and compare to the expected shape
        check_shape = np.array(
            [
                dataset.__getitem__(i)[0].shape == expected_data_shape
                for i in range(len(dataset))
            ]
        ).sum()
        # show that every image has the correct shape
        assert check_shape == len(dataset)


